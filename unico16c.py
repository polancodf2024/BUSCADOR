import streamlit as st
import pandas as pd
from datetime import datetime
import time
import requests
from xml.etree import ElementTree
import re
from collections import Counter
import numpy as np
import math
import string
import os
from io import BytesIO
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN PARA STREAMLIT CLOUD
# ============================================================================

# Configurar modelos de embeddings con fallback robusto
AI_EMBEDDINGS_AVAILABLE = False
BIOMED_EMBEDDER = None
FALLBACK_EMBEDDER = None
USE_FALLBACK = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import HDBSCAN, KMeans, AgglomerativeClustering
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    import torch
    
    # Intento 1: BioBERT (mejor para biomedicina)
    with st.spinner("🔄 Loading BioBERT (first time may take ~2 minutes)..."):
        try:
            BIOMED_EMBEDDER = SentenceTransformer(
                'pritamdeka/S-Biomed-Roberta-snli-multinli-stsb',
                device='cpu'
            )
            AI_EMBEDDINGS_AVAILABLE = True
            USE_FALLBACK = False
            st.success("✅ BioBERT embeddings available")
            print("✅ BioBERT embeddings loaded successfully")
        except Exception as e:
            print(f"⚠️ BioBERT failed to load: {e}")
            
            # Intento 2: SBERT general (fallback)
            with st.spinner("🔄 BioBERT failed, loading SBERT (general model)..."):
                try:
                    FALLBACK_EMBEDDER = SentenceTransformer(
                        'all-MiniLM-L6-v2',
                        device='cpu'
                    )
                    AI_EMBEDDINGS_AVAILABLE = True
                    USE_FALLBACK = True
                    st.warning("⚠️ BioBERT unavailable, using SBERT (general model)")
                    print("✅ SBERT (fallback) loaded successfully")
                except Exception as e2:
                    print(f"⚠️ SBERT also failed: {e2}")
                    AI_EMBEDDINGS_AVAILABLE = False
                    USE_FALLBACK = False
                    
except Exception as e:
    print(f"⚠️ Embeddings not available: {e}")
    AI_EMBEDDINGS_AVAILABLE = False
    USE_FALLBACK = False

# Configuración para TF-IDF + LSA como último recurso
TFIDF_AVAILABLE = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    TFIDF_AVAILABLE = True
except:
    TFIDF_AVAILABLE = False

# Configuración de la página
st.set_page_config(
    page_title="PubMed AI Analyzer - Advanced Flavor Generator",
    page_icon="🧠",
    layout="centered"
)

# ============================================================================
# FUNCIONES DE UTILIDAD PARA PubMed
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_abstract(pmid):
    """Get article abstract from PubMed with caching"""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    fetch_url = f"{base_url}efetch.fcgi"
    
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml",
        "rettype": "abstract"
    }
    
    try:
        response = requests.get(fetch_url, params=params, timeout=10)
        response.raise_for_status()
        
        root = ElementTree.fromstring(response.content)
        
        abstract_texts = []
        for abstract in root.findall(".//AbstractText"):
            if abstract.text:
                abstract_texts.append(abstract.text)
        
        return " ".join(abstract_texts) if abstract_texts else None
        
    except Exception as e:
        return None

def preprocess_text(text):
    """Preprocess text for NLP analysis"""
    if not text:
        return ""
    
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    
    return text

# ============================================================================
# MEJORA 1: NER CON BIOBERT PARA EXTRACCIÓN DE ENTIDADES
# ============================================================================

def get_embedder():
    """Get the available embedder (BioBERT, SBERT, or None)"""
    if BIOMED_EMBEDDER is not None:
        return BIOMED_EMBEDDER
    elif FALLBACK_EMBEDDER is not None:
        return FALLBACK_EMBEDDER
    else:
        return None

def extract_entities_with_biobert(text):
    """Extract medical entities using BioBERT embeddings or fallback"""
    embedder = get_embedder()
    if not embedder or not text:
        return []
    
    try:
        entity_list = [
            'VP40', 'VP35', 'GP', 'glycoprotein', 'nucleoprotein', 'NP',
            'Ebola virus', 'EBOV', 'Zaire ebolavirus',
            'matrix protein', 'polymerase', 'polymerase cofactor',
            'interferon antagonist', 'immune evasion', 'viral replication',
            'mortality', 'fatality', 'survival', 'bleeding', 'hemorrhage',
            'efficacy', 'safety', 'toxicity', 'inhibitor', 'vaccine',
            'antibody', 'neutralization', 'docking', 'molecular dynamics'
        ]
        
        text_embedding = embedder.encode([text[:2000]])[0]
        entity_embeddings = embedder.encode(entity_list)
        
        similarities = cosine_similarity([text_embedding], entity_embeddings)[0]
        
        # Adjust threshold based on embedder type
        threshold = 0.60 if USE_FALLBACK else 0.65
        
        entities = []
        for i, sim in enumerate(similarities):
            if sim > threshold:
                entities.append((entity_list[i], 'biobert_ner', sim))
        
        return entities
    except Exception as e:
        return []

def extract_entities_with_regex(text):
    """Extract entities using regex as backup"""
    if not text:
        return []
    
    text_lower = text.lower()
    entities = []
    
    medical_patterns = {
        'proteins': r'\b(VP40|VP35|VP30|VP24|GP|NP|nucleoprotein|glycoprotein|matrix protein|polymerase)\b',
        'medications': r'\b(ticagrelor|clopidogrel|aspirin|prasugrel|heparin|warfarin|rivaroxaban|fendiline|galidesivir)\b',
        'conditions': r'\b(ebola|ebov|ebolavirus|hemorrhagic fever)\b',
        'methods': r'\b(machine learning|deep learning|molecular dynamics|docking|simulation|bioinformatics|computational)\b',
        'outcomes': r'\b(mortality|death|survival|bleeding|hemorrhage|efficacy|safety|toxicity|effectiveness)\b',
        'study_types': r'\b(randomized|rct|cohort|prospective|retrospective|observational|meta-analysis)\b'
    }
    
    for entity_type, pattern in medical_patterns.items():
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            entities.append((match, entity_type, 1.0))
    
    return entities

def extract_medical_entities_enhanced(text):
    """Extract entities combining BioBERT NER and regex"""
    if not text:
        return []
    
    biobert_entities = extract_entities_with_biobert(text)
    regex_entities = extract_entities_with_regex(text)
    
    all_entities = []
    seen = set()
    
    for entity, etype, score in biobert_entities:
        if entity.lower() not in seen:
            all_entities.append((entity, etype, score))
            seen.add(entity.lower())
    
    for entity, etype, score in regex_entities:
        if entity.lower() not in seen:
            all_entities.append((entity, etype, score))
            seen.add(entity.lower())
    
    return all_entities

def extract_numeric_results(text):
    """Extract important numerical results"""
    if not text:
        return []
    
    results = []
    
    hr_pattern = r'(?:HR|hazard ratio)[\s:=]+([0-9]+\.[0-9]+)\s*\(([0-9]+\.[0-9]+)[-\s]+([0-9]+\.[0-9]+)\)'
    for match in re.findall(hr_pattern, text, re.IGNORECASE):
        results.append({'type': 'HR', 'value': float(match[0]), 'ci_lower': float(match[1]), 'ci_upper': float(match[2])})
    
    rr_pattern = r'(?:RR|risk ratio)[\s:=]+([0-9]+\.[0-9]+)\s*\(([0-9]+\.[0-9]+)[-\s]+([0-9]+\.[0-9]+)\)'
    for match in re.findall(rr_pattern, text, re.IGNORECASE):
        results.append({'type': 'RR', 'value': float(match[0]), 'ci_lower': float(match[1]), 'ci_upper': float(match[2])})
    
    or_pattern = r'(?:OR|odds ratio)[\s:=]+([0-9]+\.[0-9]+)\s*\(([0-9]+\.[0-9]+)[-\s]+([0-9]+\.[0-9]+)\)'
    for match in re.findall(or_pattern, text, re.IGNORECASE):
        results.append({'type': 'OR', 'value': float(match[0]), 'ci_lower': float(match[1]), 'ci_upper': float(match[2])})
    
    p_pattern = r'[Pp]\s*[=<]\s*([0-9]+\.[0-9]+)'
    for match in re.findall(p_pattern, text):
        results.append({'type': 'p-value', 'value': float(match)})
    
    return results

def extract_all_outcomes(text):
    """Extract all mentioned outcomes"""
    if not text:
        return []
    
    text_lower = text.lower()
    outcomes = []
    
    outcome_patterns = [
        r'\b(mortality|death|survival)\b',
        r'\b(bleeding|hemorrhage)\b',
        r'\b(stroke|cva|tia)\b',
        r'\b(reinfarction|mi|myocardial infarction)\b',
        r'\b(mace)\b',
        r'\b(efficacy|safety|effectiveness)\b',
        r'\b(complication|adverse event)\b',
        r'\b(risk|rate|incidence)\b',
        r'\b(outcome|endpoint)\b',
        r'\b(recovery|improvement)\b'
    ]
    
    for pattern in outcome_patterns:
        matches = re.findall(pattern, text_lower)
        outcomes.extend(matches)
    
    return list(set(outcomes))

def analyze_sentiment_by_outcome(text):
    """Analyze sentiment for each outcome with improved negation detection"""
    if not text:
        return {}
    
    text_lower = text.lower()
    outcomes_sentiment = {}
    all_outcomes = extract_all_outcomes(text)
    
    negation_words = ['no', 'not', 'without', 'lack', 'failed', 'non-significant', 'did not', 'does not']
    intensifiers = ['significantly', 'markedly', 'dramatically', 'strongly']
    
    for outcome in all_outcomes:
        outcome_index = text_lower.find(outcome)
        if outcome_index == -1:
            continue
            
        start = max(0, outcome_index - 300)
        end = min(len(text_lower), outcome_index + 300)
        context = text_lower[start:end]
        
        positive_terms = ['reduced', 'decreased', 'lower', 'improved', 'better', 'benefit', 
                         'protective', 'effective', 'efficacious', 'superior', 'advantage']
        negative_terms = ['increased', 'higher', 'worse', 'elevated', 'risk', 'adverse', 
                         'harm', 'detrimental', 'inferior', 'disadvantage']
        
        sentiment_score = 0
        
        for term in positive_terms:
            if term in context:
                term_pos = context.find(term)
                prev_words = context[max(0, term_pos-50):term_pos]
                if any(neg in prev_words for neg in negation_words):
                    sentiment_score -= 1
                else:
                    if any(inten in prev_words for inten in intensifiers):
                        sentiment_score += 2
                    else:
                        sentiment_score += 1
        
        for term in negative_terms:
            if term in context:
                term_pos = context.find(term)
                prev_words = context[max(0, term_pos-50):term_pos]
                if any(neg in prev_words for neg in negation_words):
                    sentiment_score += 1
                else:
                    if any(inten in prev_words for inten in intensifiers):
                        sentiment_score -= 2
                    else:
                        sentiment_score -= 1
        
        if sentiment_score > 0:
            outcomes_sentiment[outcome] = 'BENEFIT'
        elif sentiment_score < 0:
            outcomes_sentiment[outcome] = 'RISK'
        else:
            outcomes_sentiment[outcome] = 'NO EFFECT'
    
    return outcomes_sentiment

def enhanced_quality_scoring(study_types, full_text):
    """Enhanced quality scoring system"""
    score = 0
    factors = []
    text_lower = full_text.lower() if full_text else ""
    
    study_type_weights = {'Meta-analysis': 2.5, 'RCT': 2.0, 'Cohort': 1.6, 'Case-control': 1.4, 'Observational': 1.2}
    
    for study_type in study_types:
        if study_type in study_type_weights:
            score += study_type_weights[study_type] * 20
            factors.append(f"Study type: {study_type}")
    
    sample_patterns = [r'n\s*[=:]\s*(\d+)', r'sample size\s*[=:]\s*(\d+)']
    max_sample = 0
    for pattern in sample_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            try:
                max_sample = max(max_sample, int(match))
            except:
                pass
    
    if max_sample > 0:
        if max_sample > 1000:
            score += 20
            factors.append(f"Large sample: n={max_sample}")
        elif max_sample > 500:
            score += 15
        elif max_sample > 200:
            score += 10
        elif max_sample > 100:
            score += 5
    
    if any(term in text_lower for term in ['multicenter', 'multi-center']):
        score += 10
        factors.append("Multicenter study")
    
    if any(term in text_lower for term in ['double-blind', 'double blind', 'masked']):
        score += 15
        factors.append("Blinded")
    
    return min(score, 100), factors

def get_effect_direction(result_value, ci_lower=None, ci_upper=None):
    """Determine effect directionality"""
    try:
        value = float(result_value)
        if value < 1:
            if ci_upper and float(ci_upper) < 1:
                return "PROTECTIVE"
            else:
                return "PROTECTIVE (trend)"
        elif value > 1:
            if ci_lower and float(ci_lower) > 1:
                return "HARMFUL"
            else:
                return "HARMFUL (trend)"
        else:
            return "NO EFFECT"
    except:
        return "NOT DETERMINED"

def calculate_evidence_strength(statistical_results, quality_score):
    """Calculate evidence strength"""
    if not statistical_results:
        return "No data", 0, {}
    
    strength_score = 0
    directions = {}
    
    for i, result in enumerate(statistical_results):
        if result['type'] in ['HR', 'RR', 'OR']:
            effect = result['value']
            if effect < 0.5 or effect > 2.0:
                strength_score += 30
            elif effect < 0.7 or effect > 1.5:
                strength_score += 20
            elif effect != 1.0:
                strength_score += 10
            
            directions[f"result_{i}"] = get_effect_direction(effect, result.get('ci_lower'), result.get('ci_upper'))
        
        if result['type'] == 'p-value':
            if result['value'] < 0.001:
                strength_score += 25
            elif result['value'] < 0.01:
                strength_score += 20
            elif result['value'] < 0.05:
                strength_score += 15
    
    strength_score = strength_score * (quality_score / 50)
    
    if strength_score >= 50:
        strength = "STRONG EVIDENCE"
    elif strength_score >= 30:
        strength = "MODERATE EVIDENCE"
    elif strength_score >= 15:
        strength = "WEAK EVIDENCE"
    else:
        strength = "INSUFFICIENT EVIDENCE"
    
    return strength, min(strength_score, 100), directions

def analyze_article_with_ai(title, abstract):
    """Analyze an article with enhanced AI"""
    if not title and not abstract:
        return {}
    
    full_text = f"{title} {abstract if abstract else ''}"
    processed_text = preprocess_text(full_text)
    
    entities = extract_medical_entities_enhanced(full_text)
    
    study_types = []
    study_keywords = {
        'RCT': ['randomized', 'rct', 'randomised', 'trial', 'double-blind'],
        'Cohort': ['cohort', 'follow-up', 'longitudinal', 'prospective', 'retrospective'],
        'Meta-analysis': ['meta-analysis', 'meta analysis', 'systematic review'],
        'Observational': ['observational', 'registry', 'real-world'],
        'Case-control': ['case-control', 'matched']
    }
    
    for study_type, keywords in study_keywords.items():
        if any(keyword in processed_text for keyword in keywords):
            study_types.append(study_type)
    
    words = processed_text.split()
    word_freq = Counter(words)
    top_keywords = [word for word, _ in word_freq.most_common(20) if len(word) > 3][:10]
    
    population_patterns = {
        'adults': r'\badults?\b|\bpatients?\b',
        'elderly': r'\belderly\b|\baged\b|\bolder\b',
        'women': r'\bwomen\b|\bfemale\b',
        'men': r'\bmen\b|\bmale\b'
    }
    
    population = [pop for pop, pattern in population_patterns.items() if re.search(pattern, processed_text)]
    
    numeric_results = extract_numeric_results(full_text)
    all_outcomes = extract_all_outcomes(full_text)
    
    quality_score, quality_factors = enhanced_quality_scoring(study_types, full_text)
    
    evidence_strength, evidence_score, directions = calculate_evidence_strength(numeric_results, quality_score)
    
    sentiment = analyze_sentiment_by_outcome(full_text)
    
    numeric_results_str = ' | '.join([f"{r['type']}={r['value']}" for r in numeric_results[:3]])
    
    return {
        'entities': entities,
        'study_types': ', '.join(study_types) if study_types else 'Not specified',
        'top_keywords': ', '.join(top_keywords),
        'population': ', '.join(population) if population else 'Not specified',
        'quality_score': quality_score,
        'quality_factors': ' | '.join(quality_factors) if quality_factors else 'No factors',
        'numeric_results': numeric_results,
        'numeric_results_str': numeric_results_str,
        'outcomes_analysis': sentiment,
        'all_outcomes': all_outcomes,
        'evidence_strength': evidence_strength,
        'evidence_score': evidence_score,
        'effect_directions': directions,
        'abstract': abstract if abstract else ''
    }

def extract_article_info(doc_sum):
    """Extract basic article information from DocSum"""
    article = {}
    article["pmid"] = doc_sum.find("Id").text if doc_sum.find("Id") is not None else "N/A"
    
    title_item = doc_sum.find(".//Item[@Name='Title']")
    article["title"] = title_item.text if title_item is not None else "No title"
    
    author_items = doc_sum.findall(".//Item[@Name='Author']")
    authors = [author.text for author in author_items if author.text]
    article["authors"] = ", ".join(authors[:5]) + (" et al." if len(authors) > 5 else "")
    
    source_item = doc_sum.find(".//Item[@Name='Source']")
    article["journal"] = source_item.text if source_item is not None else "N/A"
    
    pubdate_item = doc_sum.find(".//Item[@Name='PubDate']")
    article["pubdate"] = pubdate_item.text if pubdate_item is not None else "N/A"
    
    doi_item = doc_sum.find(".//Item[@Name='DOI']")
    article["doi"] = doi_item.text if doi_item is not None else "N/A"
    
    return article

def search_pubmed(query, retmax=100000):
    """Search articles in PubMed - NOW SUPPORTS UP TO 100,000"""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    search_url = f"{base_url}esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "xml",
        "sort": "relevance"
    }
    
    try:
        response = requests.get(search_url, params=search_params, timeout=30)
        response.raise_for_status()
        
        root = ElementTree.fromstring(response.content)
        id_list = [id_elem.text for id_elem in root.findall(".//Id")]
        count = root.find(".//Count").text if root.find(".//Count") is not None else "0"
        
        return id_list, int(count)
    except Exception as e:
        st.error(f"Search error: {e}")
        return [], 0

def fetch_articles_details(id_list):
    """Fetch article details and analyze them - PROCESSES ALL ARTICLES FOUND"""
    if not id_list:
        return []
    
    total_to_process = len(id_list)
    batch_size = 50
    num_batches = math.ceil(total_to_process / batch_size)
    
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    articles = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, total_to_process)
        batch_ids = id_list[start_idx:end_idx]
        
        status_text.text(f"📦 Processing batch {batch_num + 1} of {num_batches} ({len(batch_ids)} articles)...")
        
        summary_params = {
            "db": "pubmed",
            "id": ",".join(batch_ids),
            "retmode": "xml"
        }
        
        try:
            summary_response = requests.get(f"{base_url}esummary.fcgi", params=summary_params, timeout=30)
            summary_response.raise_for_status()
            summary_root = ElementTree.fromstring(summary_response.content)
            
            for j, doc_sum in enumerate(summary_root.findall(".//DocSum")):
                overall_idx = start_idx + j
                progress_bar.progress((overall_idx + 1) / total_to_process)
                
                article = extract_article_info(doc_sum)
                abstract = get_abstract(article["pmid"])
                article["abstract"] = abstract if abstract else "Not available"
                
                ai_analysis = analyze_article_with_ai(article["title"], abstract)
                article.update(ai_analysis)
                
                articles.append(article)
                time.sleep(0.05)
                
        except Exception as e:
            st.warning(f"Error in batch {batch_num + 1}: {str(e)[:100]}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return articles

def calculate_relevance_to_search_and_hypothesis(articles, query, hypothesis):
    """Calculate relevance to search and hypothesis using embeddings"""
    embedder = get_embedder()
    if not embedder or not AI_EMBEDDINGS_AVAILABLE:
        return articles
    
    try:
        query_text = query[:500] if len(query) > 500 else query
        query_embedding = embedder.encode([query_text])[0]
        hypothesis_embedding = embedder.encode([hypothesis])[0]
        
        for article in articles:
            text = f"{article.get('title', '')} {article.get('abstract', '')}"
            if text and len(text) > 50:
                article_embedding = embedder.encode([text[:1000]])[0]
                
                search_sim = cosine_similarity([query_embedding], [article_embedding])[0][0]
                hypothesis_sim = cosine_similarity([hypothesis_embedding], [article_embedding])[0][0]
                
                article['relevance_score'] = (search_sim * 0.3) + (hypothesis_sim * 0.7)
                article['search_relevance'] = float(search_sim)
                article['hypothesis_relevance'] = float(hypothesis_sim)
            else:
                article['relevance_score'] = 0.0
                article['search_relevance'] = 0.0
                article['hypothesis_relevance'] = 0.0
        
        articles.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    except Exception as e:
        st.warning(f"Error calculating relevance: {e}")
        for article in articles:
            article['relevance_score'] = 0.5
    
    return articles

def filter_articles_by_relevance(articles, relevance_threshold):
    """Filter articles by relevance threshold"""
    filtered = [a for a in articles if a.get('relevance_score', 0) >= relevance_threshold]
    return filtered

def generate_descriptive_name(articles):
    """Generate a descriptive name for a cluster of articles"""
    if not articles:
        return "General"
    
    all_keywords = []
    all_outcomes = []
    all_proteins = []
    
    for a in articles:
        title = a.get('title', '').lower()
        keywords = a.get('top_keywords', '').split(', ')
        all_keywords.extend(keywords)
        outcomes = a.get('all_outcomes', [])
        all_outcomes.extend(outcomes)
        
        if 'vp40' in title:
            all_proteins.append('VP40')
        if 'vp35' in title:
            all_proteins.append('VP35')
        if 'gp' in title or 'glycoprotein' in title:
            all_proteins.append('GP')
        if 'np' in title or 'nucleoprotein' in title:
            all_proteins.append('NP')
        if 'vaccine' in title:
            all_proteins.append('vaccine')
        if 'machine learning' in title or 'ml' in title:
            all_proteins.append('ML')
        if 'drug' in title or 'inhibitor' in title:
            all_proteins.append('drug discovery')
        if 'structure' in title or 'dynamics' in title or 'simulation' in title:
            all_proteins.append('structure')
    
    protein_counts = Counter(all_proteins)
    outcome_counts = Counter(all_outcomes)
    
    main_topic = ''
    if protein_counts:
        main_topic = protein_counts.most_common(1)[0][0]
    
    if not main_topic and all_keywords:
        main_topic = all_keywords[0].title() if all_keywords else ''
    
    if main_topic:
        name = main_topic.upper()
    else:
        name = "Research"
    
    if outcome_counts:
        top_outcomes = [o for o, _ in outcome_counts.most_common(2)]
        if top_outcomes:
            name += f": {', '.join(top_outcomes)}"
    
    return name

def generate_citation_text(article, index):
    """Generate citation text in Vancouver format"""
    authors = article.get('authors', 'Author')
    year = article.get('pubdate', 'n.d.')[:4] if article.get('pubdate') else 'n.d.'
    title = article.get('title', 'No title')
    journal = article.get('journal', 'Journal')
    pmid = article.get('pmid', '')
    
    citation = f"{index}. {authors}. {title}. {journal}. {year}"
    if pmid and pmid != 'N/A':
        citation += f"; PMID: {pmid}"
    
    return citation

# ============================================================================
# MEJORA 3: DETERMINACIÓN AVANZADA DE ASPECTO Y DIFERENCIA CON TF-IDF + LSA
# ============================================================================

def extract_topic_keywords_tfidf(articles, n_keywords=5):
    """Extract topic keywords using TF-IDF for theme classification"""
    if not articles or len(articles) < 2:
        return []
    
    texts = []
    for a in articles[:20]:  # Limit for performance
        title = a.get('title', '')
        outcomes = ' '.join(a.get('all_outcomes', []))
        text = f"{title} {outcomes}"
        if text:
            texts.append(text)
    
    if len(texts) < 2:
        return []
    
    try:
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Get top terms across all documents
        feature_names = vectorizer.get_feature_names_out()
        
        # Sum TF-IDF scores across documents
        scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
        top_indices = scores.argsort()[-n_keywords:][::-1]
        
        top_keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
        return top_keywords[:n_keywords]
    except Exception as e:
        return []

def determine_flavor_aspect_and_difference(articles, flavor_name):
    """Enhanced aspect and difference determination using TF-IDF and rule-based logic"""
    if not articles:
        return "General analysis", "Integrative approach"
    
    # Analyze article characteristics
    all_outcomes = []
    all_methods = []
    all_proteins = []
    all_applications = []
    all_titles = []
    
    for a in articles:
        outcomes = a.get('all_outcomes', [])
        all_outcomes.extend(outcomes)
        
        text = f"{a.get('title', '')} {a.get('abstract', '')}".lower()
        all_titles.append(a.get('title', ''))
        
        if 'machine learning' in text or 'ml' in text:
            all_methods.append('machine learning')
        if 'molecular dynamics' in text or 'dynamics' in text:
            all_methods.append('molecular dynamics')
        if 'docking' in text:
            all_methods.append('docking')
        if 'virtual screening' in text:
            all_methods.append('virtual screening')
        if 'vaccine' in text:
            all_applications.append('vaccine development')
        if 'drug' in text or 'inhibitor' in text or 'therapeutic' in text:
            all_applications.append('drug discovery')
        if 'structure' in text:
            all_applications.append('structural biology')
        if 'evolution' in text or 'phylogenetic' in text:
            all_applications.append('evolutionary analysis')
        
        if 'vp40' in text:
            all_proteins.append('VP40')
        if 'vp35' in text:
            all_proteins.append('VP35')
        if 'gp' in text or 'glycoprotein' in text:
            all_proteins.append('GP')
        if 'np' in text:
            all_proteins.append('NP')
    
    # Use TF-IDF to extract topic keywords for novelty detection
    tfidf_keywords = extract_topic_keywords_tfidf(articles, n_keywords=5)
    
    outcome_counts = Counter(all_outcomes)
    method_counts = Counter(all_methods)
    protein_counts = Counter(all_proteins)
    app_counts = Counter(all_applications)
    
    # Determine aspect (main characteristic)
    aspect = ""
    if app_counts:
        top_app = app_counts.most_common(1)[0][0]
        if top_app == 'drug discovery':
            aspect = "Drug Discovery and Virtual Screening"
        elif top_app == 'vaccine development':
            aspect = "Vaccine Development and Immunoinformatics"
        elif top_app == 'structural biology':
            aspect = "Structural Biology and Molecular Dynamics"
        elif top_app == 'evolutionary analysis':
            aspect = "Phylogenetic Analysis and Viral Evolution"
        else:
            aspect = "Computational Applications"
    elif protein_counts:
        top_protein = protein_counts.most_common(1)[0][0]
        aspect = f"Studies on {top_protein} Protein"
    elif outcome_counts:
        top_outcome = outcome_counts.most_common(1)[0][0]
        aspect = f"Evaluation of {top_outcome.capitalize()}"
    elif method_counts:
        top_method = method_counts.most_common(1)[0][0]
        aspect = f"Application of {top_method.capitalize()}"
    else:
        # Use TF-IDF keywords for novel topics
        if tfidf_keywords:
            aspect = f"Emerging Topic: {', '.join(tfidf_keywords[:3])}"
        else:
            aspect = "Integrative Computational Analysis"
    
    # Determine difference (what makes this flavor unique)
    difference = ""
    
    # Check for unique methodological emphasis
    if method_counts.get('machine learning', 0) > len(articles) * 0.5:
        difference = "Predominantly uses machine learning algorithms for prediction and classification"
    elif method_counts.get('molecular dynamics', 0) > len(articles) * 0.4:
        difference = "Emphasizes molecular dynamics simulations for structural characterization"
    elif method_counts.get('docking', 0) > len(articles) * 0.4:
        difference = "Focuses on molecular docking and virtual screening for inhibitor identification"
    
    # Check for application focus
    elif app_counts.get('drug discovery', 0) > len(articles) * 0.4:
        difference = "Applied approach to drug discovery with emphasis on virtual screening and in silico validation"
    elif app_counts.get('vaccine development', 0) > len(articles) * 0.3:
        difference = "Focused on rational vaccine design through reverse vaccinology and epitope mapping"
    elif app_counts.get('structural biology', 0) > len(articles) * 0.4:
        difference = "Centered on structural characterization and protein-protein interactions"
    
    # Check for protein focus
    elif protein_counts.get('VP40', 0) > len(articles) * 0.3:
        difference = "Centered on VP40 protein and its role in viral assembly and budding"
    elif protein_counts.get('VP35', 0) > len(articles) * 0.3:
        difference = "Centered on VP35 protein and its function in replication and immune evasion"
    elif protein_counts.get('GP', 0) > len(articles) * 0.3:
        difference = "Centered on glycoprotein GP and its role in viral entry"
    elif protein_counts.get('NP', 0) > len(articles) * 0.3:
        difference = "Centered on nucleoprotein NP and its role in genome packaging"
    
    # Check for outcome focus
    elif outcome_counts.get('mortality', 0) > len(articles) * 0.3:
        difference = "Focuses on clinical outcomes of mortality and survival"
    elif outcome_counts.get('efficacy', 0) > len(articles) * 0.3:
        difference = "Quantitative assessment of efficacy and effectiveness"
    elif outcome_counts.get('safety', 0) > len(articles) * 0.3:
        difference = "Emphasis on safety profiles and adverse events"
    
    # Check for contradictory results
    elif 'mixed' in flavor_name.lower() or 'contradictory' in flavor_name.lower():
        difference = "Aggregates studies with heterogeneous or contradictory findings"
    
    # Check for population focus
    elif 'population' in flavor_name.lower():
        difference = "Focuses on specific populations with distinct demographic characteristics"
    
    # Use TF-IDF keywords for novel topics
    elif tfidf_keywords:
        difference = f"Explores emerging themes including {', '.join(tfidf_keywords[:3])}"
    
    else:
        difference = "Integrates studies with diverse methodologies and thematic approaches"
    
    # Add embedder info for context
    if USE_FALLBACK:
        difference += " (Analysis performed with general SBERT model)"
    elif not AI_EMBEDDINGS_AVAILABLE:
        difference += " (Analysis performed with TF-IDF-based methods)"
    
    return aspect, difference

# ============================================================================
# MEJORA 4: MANEJO DE DUPLICADOS CON ASIGNACIÓN EXCLUSIVA
# ============================================================================

def assign_articles_to_best_flavor(flavors):
    """Assign each article to the most relevant flavor (exclusive assignment)"""
    if not flavors:
        return []
    
    # Create a mapping of article IDs to their best flavor
    article_to_flavor = {}
    article_scores = {}
    
    # For each flavor, score each article
    for flavor in flavors:
        flavor_id = flavor['id']
        flavor_articles = flavor['articles']
        
        # Calculate a relevance score for each article in this flavor
        # Use quality score and title relevance as proxy
        for article in flavor_articles:
            article_id = article.get('pmid', id(article))
            score = article.get('quality_score', 50) / 100.0
            
            # Boost score if title matches flavor theme
            flavor_name_lower = flavor['name'].lower()
            title_lower = article.get('title', '').lower()
            if any(keyword in title_lower for keyword in flavor_name_lower.split()[:3]):
                score += 0.2
            
            if article_id not in article_scores or score > article_scores[article_id]:
                article_scores[article_id] = score
                article_to_flavor[article_id] = flavor_id
    
    # Rebuild flavors with exclusive article assignment
    flavor_article_map = {flavor['id']: [] for flavor in flavors}
    for article in [a for flavor in flavors for a in flavor['articles']]:
        article_id = article.get('pmid', id(article))
        if article_id in article_to_flavor:
            assigned_flavor = article_to_flavor[article_id]
            if article not in flavor_article_map[assigned_flavor]:
                flavor_article_map[assigned_flavor].append(article)
    
    # Update flavors with exclusive articles
    for flavor in flavors:
        flavor['articles'] = flavor_article_map.get(flavor['id'], [])
        flavor['n_articles'] = len(flavor['articles'])
        # Update representative articles
        flavor['representative_articles'] = sorted(
            flavor['articles'], 
            key=lambda x: x.get('quality_score', 0), 
            reverse=True
        )[:5]
    
    # Remove empty flavors
    flavors = [f for f in flavors if f['n_articles'] >= 2]
    
    return flavors

# ============================================================================
# FUNCIONES DE CLUSTERING MEJORADAS
# ============================================================================

def get_text_embeddings(texts):
    """Get embeddings with fallback to TF-IDF if needed"""
    embedder = get_embedder()
    
    if embedder and AI_EMBEDDINGS_AVAILABLE:
        try:
            return embedder.encode(texts)
        except:
            pass
    
    # Fallback: TF-IDF + TruncatedSVD
    if TFIDF_AVAILABLE:
        try:
            vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            svd = TruncatedSVD(n_components=min(50, tfidf_matrix.shape[1] - 1), random_state=42)
            return svd.fit_transform(tfidf_matrix)
        except:
            pass
    
    return None

def discover_flavors_by_embeddings_hdbscan(articles):
    """Discover flavors using HDBSCAN with embedding fallback"""
    if len(articles) < 3:
        return []
    
    try:
        texts = [f"{a.get('title', '')} {' '.join(a.get('all_outcomes', []))}" for a in articles]
        embeddings = get_text_embeddings(texts)
        
        if embeddings is None:
            return []
        
        # PCA for dimensionality reduction
        n_components = min(50, embeddings.shape[1], len(embeddings) - 1)
        if n_components > 1:
            pca = PCA(n_components=n_components)
            embeddings_reduced = pca.fit_transform(embeddings)
        else:
            embeddings_reduced = embeddings
        
        clusterer = HDBSCAN(min_cluster_size=3, min_samples=2, metric='euclidean')
        cluster_labels = clusterer.fit_predict(embeddings_reduced)
        
        flavors = []
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:
                continue
            
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            if len(cluster_indices) < 3:
                continue
            
            cluster_articles = [articles[i] for i in cluster_indices]
            name = generate_descriptive_name(cluster_articles)
            representative = sorted(cluster_articles, key=lambda x: x.get('quality_score', 0), reverse=True)[:5]
            
            flavors.append({
                'type': 'semantic',
                'id': f"semantic_cluster_{cluster_id}",
                'name': name,
                'articles': cluster_articles,
                'n_articles': len(cluster_articles),
                'representative_articles': representative
            })
        
        return flavors
    except Exception as e:
        st.warning(f"Error in HDBSCAN clustering: {e}")
        return []

def discover_flavors_by_outcomes(articles):
    """Group articles by shared outcomes"""
    if len(articles) < 3:
        return []
    
    try:
        all_outcomes = set()
        for article in articles:
            outcomes = article.get('all_outcomes', [])
            all_outcomes.update(outcomes)
        
        if len(all_outcomes) < 2:
            return []
        
        all_outcomes_list = list(all_outcomes)
        outcome_matrix = []
        for article in articles:
            outcomes = set(article.get('all_outcomes', []))
            row = [1 if o in outcomes else 0 for o in all_outcomes_list]
            outcome_matrix.append(row)
        
        n_clusters = min(max(3, len(articles) // 15), 4)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(outcome_matrix)
        
        flavors = []
        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
            if len(cluster_indices) < 3:
                continue
            
            cluster_articles = [articles[i] for i in cluster_indices]
            
            outcome_counts = Counter()
            for article in cluster_articles:
                outcomes = article.get('all_outcomes', [])
                outcome_counts.update(outcomes)
            
            top_outcomes = [o for o, _ in outcome_counts.most_common(3)]
            name = f"Studies on {', '.join(top_outcomes)}"
            representative = sorted(cluster_articles, key=lambda x: x.get('quality_score', 0), reverse=True)[:5]
            
            flavors.append({
                'type': 'outcome_based',
                'id': f"outcome_cluster_{cluster_id}",
                'name': name,
                'articles': cluster_articles,
                'n_articles': len(cluster_articles),
                'representative_articles': representative
            })
        
        return flavors
    except Exception as e:
        st.warning(f"Error in outcome clustering: {e}")
        return []

def discover_flavors_by_effect_direction(articles):
    """Group articles by effect direction"""
    flavors = []
    
    direction_groups = {
        'beneficial': [],
        'harmful': [],
        'no_effect': [],
        'contradictory': []
    }
    
    for article in articles:
        directions = article.get('effect_directions', {})
        direction_values = list(directions.values()) if directions else []
        direction_str = ' '.join(direction_values).lower()
        
        if 'protective' in direction_str and 'harmful' not in direction_str:
            direction_groups['beneficial'].append(article)
        elif 'harmful' in direction_str and 'protective' not in direction_str:
            direction_groups['harmful'].append(article)
        elif 'no effect' in direction_str:
            direction_groups['no_effect'].append(article)
        else:
            direction_groups['contradictory'].append(article)
    
    direction_names = {
        'beneficial': 'Beneficial Effects',
        'harmful': 'Risk Factors',
        'no_effect': 'No Significant Effect',
        'contradictory': 'Mixed / Contradictory Results'
    }
    
    for key, group_articles in direction_groups.items():
        if len(group_articles) >= 3:
            representative = sorted(group_articles, key=lambda x: x.get('quality_score', 0), reverse=True)[:5]
            
            flavors.append({
                'type': 'effect_direction',
                'id': f"effect_{key}",
                'name': direction_names[key],
                'articles': group_articles,
                'n_articles': len(group_articles),
                'representative_articles': representative
            })
    
    return flavors

def discover_flavors_by_population(articles):
    """Group articles by studied population"""
    population_patterns = {
        'elderly': ['elderly', 'aged', 'older', 'geriatric', '>75'],
        'diabetes': ['diabetes', 'diabetic', 'dm'],
        'women': ['women', 'female'],
        'men': ['men', 'male'],
        'children': ['children', 'pediatric', 'paediatric', 'infant']
    }
    
    flavors = []
    
    for pop, patterns in population_patterns.items():
        pop_articles = []
        for article in articles:
            text = f"{article.get('title', '')} {article.get('abstract', '')}".lower()
            if any(pattern in text for pattern in patterns):
                pop_articles.append(article)
        
        if len(pop_articles) >= 3:
            pop_names = {
                'elderly': 'Elderly Population',
                'diabetes': 'Diabetic Patients',
                'women': 'Female Population',
                'men': 'Male Population',
                'children': 'Pediatric Population'
            }
            
            representative = sorted(pop_articles, key=lambda x: x.get('quality_score', 0), reverse=True)[:5]
            
            flavors.append({
                'type': 'population',
                'id': f"pop_{pop}",
                'name': pop_names.get(pop, pop),
                'articles': pop_articles,
                'n_articles': len(pop_articles),
                'representative_articles': representative
            })
    
    return flavors

def merge_small_clusters(flavors, target_count=4):
    """Merge small clusters to achieve approximately target_count large flavors"""
    if not flavors:
        return []
    
    if len(flavors) <= target_count:
        return flavors
    
    # Sort by number of articles (descending)
    flavors_sorted = sorted(flavors, key=lambda x: x['n_articles'], reverse=True)
    
    # Take first target_count-1 as base
    merged_flavors = flavors_sorted[:target_count-1]
    
    # Merge remaining articles into last flavor
    remaining_articles = []
    for flavor in flavors_sorted[target_count-1:]:
        remaining_articles.extend(flavor['articles'])
    
    if remaining_articles:
        # Generate name for merged flavor
        all_topics = []
        for article in remaining_articles:
            title = article.get('title', '').lower()
            if 'vp40' in title:
                all_topics.append('VP40')
            if 'vp35' in title:
                all_topics.append('VP35')
            if 'gp' in title:
                all_topics.append('GP')
            if 'vaccine' in title:
                all_topics.append('vaccine')
            if 'drug' in title or 'inhibitor' in title:
                all_topics.append('drug discovery')
        
        topic_counts = Counter(all_topics)
        if topic_counts:
            main_topic = topic_counts.most_common(1)[0][0].upper()
            name = f"Additional Studies on {main_topic}"
        else:
            name = "Additional Computational Studies"
        
        # Extract main outcomes from merged group
        all_outcomes = []
        for article in remaining_articles:
            all_outcomes.extend(article.get('all_outcomes', []))
        outcome_counts = Counter(all_outcomes)
        if outcome_counts:
            top_outcomes = [o for o, _ in outcome_counts.most_common(2)]
            if top_outcomes:
                name += f": {', '.join(top_outcomes)}"
        
        representative = sorted(remaining_articles, key=lambda x: x.get('quality_score', 0), reverse=True)[:5]
        
        merged_flavors.append({
            'type': 'merged',
            'id': 'merged_cluster',
            'name': name,
            'articles': remaining_articles,
            'n_articles': len(remaining_articles),
            'representative_articles': representative
        })
    
    return merged_flavors

def generate_all_flavors(articles, hypothesis):
    """Generate all flavors from multiple perspectives and merge them"""
    if not articles:
        return {}
    
    all_flavors = []
    
    # Generate clusters from different perspectives
    if len(articles) >= 5 and AI_EMBEDDINGS_AVAILABLE:
        semantic_flavors = discover_flavors_by_embeddings_hdbscan(articles)
        all_flavors.extend(semantic_flavors)
    
    if len(articles) >= 5:
        outcome_flavors = discover_flavors_by_outcomes(articles)
        all_flavors.extend(outcome_flavors)
        effect_flavors = discover_flavors_by_effect_direction(articles)
        all_flavors.extend(effect_flavors)
        population_flavors = discover_flavors_by_population(articles)
        all_flavors.extend(population_flavors)
    
    # If no clusters, create one with all articles
    if not all_flavors:
        all_flavors = [{
            'type': 'default',
            'id': 'default_cluster',
            'name': 'Computational Studies on Ebola Virus',
            'articles': articles,
            'n_articles': len(articles),
            'representative_articles': articles[:5]
        }]
    
    # Assign articles to best flavor (exclusive assignment)
    all_flavors = assign_articles_to_best_flavor(all_flavors)
    
    # Merge small clusters to get 3-4 large flavors
    merged_flavors = merge_small_clusters(all_flavors, target_count=4)
    
    # Organize by type for document structure
    flavors_by_category = {
        'semantic_clusters': [],
        'outcome_clusters': [],
        'effect_clusters': [],
        'population_clusters': [],
        'merged_clusters': []
    }
    
    for flavor in merged_flavors:
        category = flavor.get('type', 'semantic')
        if category == 'semantic':
            flavors_by_category['semantic_clusters'].append(flavor)
        elif category == 'outcome_based':
            flavors_by_category['outcome_clusters'].append(flavor)
        elif category == 'effect_direction':
            flavors_by_category['effect_clusters'].append(flavor)
        elif category == 'population':
            flavors_by_category['population_clusters'].append(flavor)
        else:
            flavors_by_category['merged_clusters'].append(flavor)
    
    return flavors_by_category

def generate_flavor_summary_with_citations(articles, flavor_name, section='introduction'):
    """Generate a summary paragraph with citations inserted in the text"""
    if not articles:
        return "No articles available for this flavor.", []
    
    # Determine Aspect and Difference
    aspect, difference = determine_flavor_aspect_and_difference(articles, flavor_name)
    
    # Extract key information
    main_topics = []
    main_outcomes = []
    key_findings = []
    study_types = []
    key_articles = []
    
    for idx, a in enumerate(articles[:10], 1):
        title = a.get('title', '')
        outcomes = a.get('all_outcomes', [])
        study_type = a.get('study_types', '')
        numeric = a.get('numeric_results_str', '')
        authors = a.get('authors', '').split(',')[0] if a.get('authors') else 'Author'
        
        if 'VP40' in title:
            main_topics.append('VP40')
            key_articles.append((idx, authors, 'VP40'))
        if 'VP35' in title:
            main_topics.append('VP35')
            key_articles.append((idx, authors, 'VP35'))
        if 'GP' in title or 'glycoprotein' in title:
            main_topics.append('glycoprotein')
            key_articles.append((idx, authors, 'GP'))
        if 'vaccine' in title.lower():
            main_topics.append('vaccine')
            key_articles.append((idx, authors, 'vaccine'))
        if 'drug' in title.lower() or 'inhibitor' in title.lower():
            main_topics.append('drug discovery')
            key_articles.append((idx, authors, 'drug discovery'))
        
        main_outcomes.extend(outcomes[:2])
        
        if numeric:
            key_findings.append((idx, authors, numeric))
        
        if study_type:
            study_types.append(study_type)
    
    topic_counts = Counter(main_topics)
    main_topic = topic_counts.most_common(1)[0][0] if topic_counts else 'research'
    outcome_counts = Counter(main_outcomes)
    top_outcomes = [o for o, _ in outcome_counts.most_common(3)]
    
    study_type_counts = Counter(study_types)
    top_study_types = [st for st, _ in study_type_counts.most_common(2)]
    
    # Generate header with Aspect and Difference
    header = f"### {flavor_name}\n\n"
    header += f"**🎯 Aspect:** {aspect}\n\n"
    header += f"**🔍 Difference:** {difference}\n\n"
    header += "---\n\n"
    
    # Generate paragraph based on section
    if section == 'introduction':
        summary = header
        
        if 'VP40' in main_topic or 'VP35' in main_topic or 'GP' in main_topic:
            summary += f"Bioinformatic characterization of Ebola virus proteins, particularly {main_topic}, has been addressed by {len(articles)} computational studies employing advanced techniques including machine learning, molecular dynamics, and docking. "
            summary += f"These studies have revealed critical insights into structural dynamics, protein-protein interactions, and therapeutic target potential. "
            
            if key_articles:
                refs = [f"{auth} ({idx})" for idx, auth, _ in key_articles[:5]]
                if refs:
                    summary += f"Key contributions from {', '.join(refs)} have significantly advanced understanding of {main_topic} function. "
            
            if top_outcomes:
                summary += f"Primary outcomes evaluated include {', '.join(top_outcomes[:3])}, "
                summary += f"demonstrating the clinical relevance of these in silico investigations. "
            
            if key_findings:
                find_refs = [f"{auth} ({idx})" for idx, auth, _ in key_findings[:3]]
                if find_refs:
                    summary += f"Notable findings from {', '.join(find_refs)} have identified {key_findings[0][2] if key_findings else 'significant computational predictions'}, "
                    summary += f"establishing foundations for experimental validation."
            
            summary += f" The convergence of these {len(articles)} studies demonstrates the power of computational approaches for characterizing viral proteins and identifying potential inhibitors."
        
        elif 'vaccine' in main_topic.lower():
            summary += f"Vaccine development against Ebola virus has been significantly advanced through computational approaches. This flavor integrates {len(articles)} studies employing bioinformatic methods to predict immunogenicity, identify conserved epitopes, and evaluate vaccine candidates. "
            summary += f"Reverse vaccinology and structural modeling have enabled design of constructs with higher probability of experimental success. "
            
            if key_articles:
                refs = [f"{auth} ({idx})" for idx, auth, _ in key_articles[:5]]
                if refs:
                    summary += f"Pioneering work by {', '.join(refs)} has established computational frameworks for rational vaccine design. "
            
            if top_outcomes:
                summary += f"Key outcomes assessed in this grouping include {', '.join(top_outcomes[:3])}, "
                summary += f"providing quantitative metrics for candidate selection. "
            
            summary += f"Integration of {len(articles)} independent studies strengthens evidence for identified vaccine candidates and underscores bioinformatics' role in accelerating Ebola virus vaccine design."
        
        elif 'drug' in main_topic.lower() or 'discovery' in main_topic.lower():
            summary += f"Drug discovery efforts against Ebola virus have been transformed by computational approaches. This flavor groups {len(articles)} studies employing virtual screening, molecular docking, molecular dynamics, and machine learning to identify promising compounds targeting key viral proteins. "
            summary += f"Drug repurposing and novel compound identification have emerged as complementary strategies to accelerate therapeutic development. "
            
            if key_articles:
                refs = [f"{auth} ({idx})" for idx, auth, _ in key_articles[:5]]
                if refs:
                    summary += f"Notable screening campaigns by {', '.join(refs)} have identified inhibitors with high binding affinity. "
            
            if top_outcomes:
                summary += f"Efficacy, safety, and toxicity have been primary outcomes evaluated in these {len(articles)} studies, "
                summary += f"establishing frameworks for candidate prioritization. "
            
            summary += f"Consistency across these computational studies suggests robust targets for experimental validation and drug repurposing strategies."
        
        else:
            summary += f"This flavor integrates {len(articles)} computational and bioinformatic studies investigating various aspects of Ebola virus biology, protein function, and therapeutic interventions. "
            summary += f"Methodological approaches include machine learning, molecular dynamics, molecular docking, and phylogenetic analysis. "
            
            if key_articles:
                refs = [f"{auth} ({idx})" for idx, auth, _ in key_articles[:5]]
                if refs:
                    summary += f"Key investigations by {', '.join(refs)} have advanced fundamental understanding of viral proteins. "
            
            if top_outcomes:
                summary += f"Primary outcomes examined include {', '.join(top_outcomes[:3])}, "
                summary += f"reflecting focus on clinically relevant results. "
            
            summary += f"Diversity of computational methods employed highlights both strengths and limitations of current bioinformatic approaches in studying emerging pathogens."
    
    else:  # section == 'discussion'
        summary = header
        
        if 'VP40' in main_topic or 'VP35' in main_topic or 'GP' in main_topic:
            summary += f"Our findings on {main_topic} are consistent with those reported in {len(articles)} previous computational studies that have elucidated key structural features and functional domains. "
            summary += f"The observed methodological convergence, particularly in molecular dynamics and docking, has enabled identification of conserved regions with high therapeutic value. "
            
            if key_articles:
                refs = [f"{auth} ({idx})" for idx, auth, _ in key_articles[:5]]
                if refs:
                    summary += f"Specifically, {', '.join(refs)} reported consistent results regarding {main_topic} structural dynamics, "
                    summary += f"validating computational predictions with complementary experimental data. "
            
            if key_findings:
                find_refs = [f"{auth} ({idx})" for idx, auth, _ in key_findings[:3]]
                if find_refs:
                    summary += f"Quantitative findings from {', '.join(find_refs)} provide additional evidence supporting our conclusions on the functional importance of {main_topic}. "
            
            summary += f"However, heterogeneity in computational approaches across these {len(articles)} studies suggests that method standardization could improve reproducibility and comparability. "
            summary += f"Integration of these findings with experimental structural data will be crucial for advancing therapeutic applications."
        
        elif 'vaccine' in main_topic.lower():
            summary += f"Our results are consistent with computational vaccine development studies that have identified conserved epitopes in Ebola virus. This flavor includes {len(articles)} investigations using reverse vaccinology and immunoinformatic modeling. "
            summary += f"Prediction of T and B cell epitopes has been validated through molecular docking and immunological complex dynamics. "
            
            if key_articles:
                refs = [f"{auth} ({idx})" for idx, auth, _ in key_articles[:5]]
                if refs:
                    summary += f"In particular, {', '.join(refs)} demonstrated similar immunogenicity predictions, "
                    summary += f"with high conservation across viral strains. "
            
            summary += f"Convergence of findings across {len(articles)} independent computational studies strengthens evidence for identified vaccine candidates, "
            summary += f"providing a robust framework for preclinical experimental validation and eventual clinical translation."
        
        elif 'drug' in main_topic.lower() or 'discovery' in main_topic.lower():
            summary += f"Our drug discovery findings align with those reported in {len(articles)} in silico screening studies that have identified potential inhibitors against Ebola virus proteins. "
            summary += f"Structure-based approaches have enabled identification of conserved binding sites with high pharmacological relevance. "
            
            if key_articles:
                refs = [f"{auth} ({idx})" for idx, auth, _ in key_articles[:5]]
                if refs:
                    summary += f"Notably, {', '.join(refs)} reported similar binding affinities and molecular interaction patterns, "
                    summary += f"suggesting robust targets for therapeutic development. "
            
            summary += f"Consistency across these {len(articles)} computational studies suggests robust targets for experimental validation. "
            summary += f"FDA-approved drug repurposing has emerged as a complementary strategy that could accelerate clinical development, reducing associated time and costs."
        
        else:
            summary += f"Our analysis integrates findings from {len(articles)} computational studies that have investigated Ebola virus biology through diverse in silico approaches. "
            summary += f"Diversity of computational methods employed highlights both strengths and limitations of current bioinformatic approaches. "
            summary += f"Molecular dynamics, docking, and machine learning studies have provided complementary insights into viral function. "
            
            if key_articles:
                refs = [f"{auth} ({idx})" for idx, auth, _ in key_articles[:5]]
                if refs:
                    summary += f"Key contributions from {', '.join(refs)} provide context for our findings, "
                    summary += f"establishing foundations for future investigations. "
            
            summary += f"Integration of computational data with experimental evidence will be essential for validating predictions and translating bioinformatic knowledge into effective clinical applications against Ebola virus."
    
    # Generate citations
    citations = []
    for i, article in enumerate(articles[:15], 1):
        citations.append(generate_citation_text(article, i))
    
    return summary, citations

def create_document_with_flavors(flavors, hypothesis, query, total_articles, relevance_threshold):
    """Create a DOCX document with flavors separated into Introduction and Discussion"""
    doc = Document()
    
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
    
    title = doc.add_heading('Thematic Flavors for Scientific Article', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph(f'Search strategy: {query[:200]}...')
    doc.add_paragraph(f'Hypothesis: "{hypothesis[:200]}..."')
    doc.add_paragraph(f'Total articles analyzed: {total_articles}')
    doc.add_paragraph(f'Relevance threshold applied: {relevance_threshold}')
    doc.add_paragraph(f'Generation date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    
    # Add embedder info
    if USE_FALLBACK:
        doc.add_paragraph(f'Embedding model: SBERT (fallback - BioBERT unavailable)')
    elif AI_EMBEDDINGS_AVAILABLE:
        doc.add_paragraph(f'Embedding model: BioBERT (biomedical optimized)')
    else:
        doc.add_paragraph(f'Embedding model: TF-IDF + LSA (fallback)')
    doc.add_paragraph()
    
    # ========================================================================
    # INTRODUCTION SECTION
    # ========================================================================
    doc.add_heading('FLAVORS FOR INTRODUCTION', level=1)
    doc.add_paragraph('The following paragraphs are designed for the Introduction section. They present the state of the art and justify the research.')
    doc.add_paragraph()
    
    for category_name, flavor_list in flavors.items():
        if not flavor_list:
            continue
        
        category_title = category_name.replace('_', ' ').title()
        doc.add_heading(category_title, level=2)
        
        for flavor in flavor_list:
            intro_summary, intro_citations = generate_flavor_summary_with_citations(
                flavor.get('representative_articles', flavor['articles'][:10]), 
                flavor['name'], 
                section='introduction'
            )
            
            doc.add_paragraph(intro_summary)
            doc.add_paragraph()
            doc.add_paragraph('References:', style='List Bullet')
            for citation in intro_citations:
                doc.add_paragraph(citation, style='List Bullet')
            doc.add_paragraph()
    
    doc.add_page_break()
    
    # ========================================================================
    # DISCUSSION SECTION
    # ========================================================================
    doc.add_heading('FLAVORS FOR DISCUSSION', level=1)
    doc.add_paragraph('The following paragraphs are designed for the Discussion section. They compare results with the literature and contextualize findings.')
    doc.add_paragraph()
    
    for category_name, flavor_list in flavors.items():
        if not flavor_list:
            continue
        
        category_title = category_name.replace('_', ' ').title()
        doc.add_heading(category_title, level=2)
        
        for flavor in flavor_list:
            disc_summary, disc_citations = generate_flavor_summary_with_citations(
                flavor.get('representative_articles', flavor['articles'][:10]), 
                flavor['name'], 
                section='discussion'
            )
            
            doc.add_paragraph(disc_summary)
            doc.add_paragraph()
            doc.add_paragraph('References:', style='List Bullet')
            for citation in disc_citations:
                doc.add_paragraph(citation, style='List Bullet')
            doc.add_paragraph()
    
    return doc

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.title("🧠 PubMed AI Analyzer - Advanced Flavor Generator")
    
    # Display embedder status
    if USE_FALLBACK:
        st.warning("⚠️ BioBERT unavailable - Using SBERT (general model). Semantic quality may be slightly reduced.")
    elif AI_EMBEDDINGS_AVAILABLE:
        st.success("✅ BioBERT biomedical embeddings available")
    else:
        st.warning("⚠️ Advanced embeddings unavailable - Using TF-IDF based methods")
    
    st.info("⚡ Enhanced version: All articles found | Merged flavors (3-4 large groups) | Exclusive article assignment | Aspect + Difference per flavor")
    
    st.markdown("""
    ### Generate thematic paragraphs (flavors) for your scientific article
    
    **Features:**
    - 🔍 PubMed search with embedding-based filtering (30% search + 70% hypothesis)
    - 🧬 **BioBERT/SBERT NER**: Semantic extraction of medical entities with robust fallback
    - 📊 **Clustering + merging**: Generates 3-4 large flavors with 10-15 references each
    - 📝 **Extended summaries with embedded citations**: Paragraphs of 5-7+ lines
    - 🎯 **Aspect + Difference**: Each flavor includes header with main characteristic and differential value
    - 📚 **Introduction/Discussion separation**: Section-specific paragraphs
    - ⚙️ **Configurable relevance threshold**: Adjust filtering sensitivity
    - 🔄 **Exclusive article assignment**: No article appears in multiple flavors
    - 📈 **All articles found**: No artificial limit, processes complete search results
    - 🌐 **English output**: All content generated in English
    """)
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if AI_EMBEDDINGS_AVAILABLE:
            model_name = "BioBERT" if not USE_FALLBACK else "SBERT"
            st.success(f"✅ {model_name} available")
        else:
            st.warning("⚠️ TF-IDF mode")
    with col_b:
        st.info("📄 Output: DOCX with summaries")
    with col_c:
        st.success("📈 Max: All articles found")
    
    st.markdown("---")
    st.markdown("### 📝 Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_area(
            "**PubMed search strategy:**",
            value="(((\"Ebolavirus\"[Mesh]) OR (EBOV)) AND ((\"Computational Biology\"[Mesh]) OR (\"Machine Learning\"[Mesh]) OR (bioinformatics))) AND ((\"Viral Proteins\"[Mesh]) OR (VP40 OR VP35 OR GP)) NOT (review[pt])",
            height=100,
            help="Use MeSH syntax for better results"
        )
    
    with col2:
        st.info("""
        **Article limit:**
        - All articles found in PubMed will be processed
        - No maximum limit set
        - Processing time depends on result size
        """)
    
    st.markdown("### ⚙️ Relevance filtering")
    col_rel1, col_rel2 = st.columns(2)
    
    with col_rel1:
        relevance_threshold = st.slider(
            "Relevance threshold (0 = less selective, 1 = more selective):",
            min_value=0.2,
            max_value=0.8,
            value=0.35,
            step=0.05,
            help="Lower values include more articles. 0.35 is a good balance."
        )
    
    with col_rel2:
        st.info(f"""
        **Threshold effect:**
        - {relevance_threshold:.2f}: Current
        - 0.20-0.30: Includes ~70-80% of articles
        - 0.35-0.45: Includes ~40-60% of articles (recommended)
        - 0.50+: Includes ~20-30% of articles (very selective)
        """)
    
    hypothesis = st.text_area(
        "**📌 Hypothesis (natural language):**",
        value="Machine learning-based bioinformatic characterization of Ebola virus proteins (GP, VP40, VP35, VP30, VP24, and NP) enables the accurate prediction of protein function, identification of conserved domains, and discovery of potential druggable sites for antiviral therapeutics.",
        height=100,
        help="Write your hypothesis in natural language. The program will use it to prioritize articles."
    )
    
    generate_button = st.button("🚀 GENERATE FLAVORS", type="primary", use_container_width=True)
    
    if 'docx_generated' not in st.session_state:
        st.session_state.docx_generated = False
        st.session_state.docx_data = None
        st.session_state.total_articles = 0
        st.session_state.total_processed = 0
        st.session_state.n_flavors = 0
    
    if generate_button:
        if not query.strip():
            st.warning("⚠️ Please enter a search strategy")
        elif not hypothesis.strip():
            st.warning("⚠️ Please enter your hypothesis")
        else:
            start_time = time.time()
            
            with st.spinner("🔍 Searching articles in PubMed..."):
                id_list, total_count = search_pubmed(query.strip(), retmax=100000)
                
                if not id_list:
                    st.error("❌ No articles found")
                    st.stop()
                
                st.info(f"📊 Found {total_count} articles. Processing all {len(id_list)} articles...")
                articles = fetch_articles_details(id_list)
            
            if not articles:
                st.error("❌ Could not process any articles")
                st.stop()
            
            st.success(f"✅ Processed {len(articles)} articles")
            
            with st.spinner("🧠 Calculating relevance with search and hypothesis (embeddings)..."):
                articles = calculate_relevance_to_search_and_hypothesis(articles, query, hypothesis)
                
                scores = [a.get('relevance_score', 0) for a in articles]
                if scores:
                    st.info(f"📊 Relevance distribution: min={min(scores):.2f}, max={max(scores):.2f}, mean={np.mean(scores):.2f}")
                
                filtered_articles = filter_articles_by_relevance(articles, relevance_threshold)
                
                st.info(f"📌 {len(filtered_articles)} articles with relevance ≥ {relevance_threshold} (out of {len(articles)} processed)")
                
                if len(filtered_articles) < 10:
                    st.warning(f"⚠️ Only {len(filtered_articles)} articles meet threshold. Consider reducing relevance threshold.")
                    
                    if len(filtered_articles) < 5 and len(articles) >= 10:
                        st.info("💡 Using all processed articles to generate flavors...")
                        filtered_articles = articles
                
                articles = filtered_articles
            
            if len(articles) < 5:
                st.error("❌ Not enough articles to generate flavors (minimum 5)")
                st.stop()
            
            with st.spinner("🔍 Discovering and merging flavors (3-4 large groups)..."):
                flavors = generate_all_flavors(articles, hypothesis)
            
            total_flavors = sum(len(flavor_list) for flavor_list in flavors.values())
            st.success(f"✅ Generated {total_flavors} large flavors from {len(articles)} articles")
            
            with st.spinner("📄 Creating document with extended summaries and embedded citations..."):
                doc = create_document_with_flavors(flavors, hypothesis, query, len(articles), relevance_threshold)
                
                docx_bytes = BytesIO()
                doc.save(docx_bytes)
                docx_bytes.seek(0)
                
                st.session_state.docx_data = docx_bytes
                st.session_state.docx_generated = True
                st.session_state.total_articles = len(articles)
                st.session_state.total_processed = len(id_list) if id_list else 0
                st.session_state.n_flavors = total_flavors
            
            elapsed_time = time.time() - start_time
            st.info(f"⏱️ Total time: {elapsed_time/60:.1f} minutes")
    
    if st.session_state.docx_generated and st.session_state.docx_data is not None:
        st.markdown("---")
        st.markdown("### 📥 Generated Document")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Articles found", st.session_state.total_processed)
        with col2:
            st.metric("Relevant articles", st.session_state.total_articles)
        with col3:
            st.metric("Large flavors", st.session_state.n_flavors)
        
        st.download_button(
            label="📥 DOWNLOAD FLAVORS (INTRO + DISCUSSION)",
            data=st.session_state.docx_data,
            file_name=f"flavors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
        
        with st.expander("📖 Usage Guide"):
            st.markdown(f"""
            **Execution Summary:**
            - Articles found in PubMed: {st.session_state.total_processed}
            - Articles with relevance ≥ {relevance_threshold}: {st.session_state.total_articles}
            - Large flavors generated: {st.session_state.n_flavors}
            
            **How to adjust results:**
            - If you get **too few articles** (<15), **reduce the relevance threshold** (e.g., 0.25-0.30)
            - If you get **too many articles** (>200), **increase the threshold** (e.g., 0.45-0.50)
            - If processing is **too slow**, consider narrowing your search query
            
            **Document Structure:**
            1. **FLAVORS FOR INTRODUCTION**: Extended paragraphs (5+ lines) with embedded citations
            2. **FLAVORS FOR DISCUSSION**: Extended paragraphs (5+ lines) with embedded citations
            
            **New Features:**
            - **🎯 Aspect**: The main characteristic that defines this group of studies
            - **🔍 Difference**: What makes this flavor unique compared to others
            - **Exclusive assignment**: Each article appears in only one flavor
            - **Robust fallback**: BioBERT → SBERT → TF-IDF automatic fallback chain
            - **English output**: All content generated in English for international use
            - **All articles processed**: No artificial limit on number of articles
            
            These elements allow quick identification of each flavor's value proposition.
            """)
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 0.8em;'>
            🧠 PubMed AI Analyzer - Advanced Flavor Generator v12.0<br>
            BioBERT → SBERT → TF-IDF Fallback | Exclusive Assignment | Aspect + Difference per Flavor<br>
            All articles processed | 3-4 large flavors with ~15 references each | English output
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
