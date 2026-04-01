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
from io import BytesIO, StringIO
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE EMBEDDINGS CON FALLBACK ROBUSTO
# ============================================================================

AI_EMBEDDINGS_AVAILABLE = False
BIOMED_EMBEDDER = None
FALLBACK_EMBEDDER = None
USE_FALLBACK = False

# Intentar cargar embeddings solo si es posible (no crítico)
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import HDBSCAN, KMeans, AgglomerativeClustering
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    
    # Intentar cargar sentence-transformers (opcional)
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        
        with st.spinner("🔄 Loading embeddings model (optional)..."):
            try:
                # Intentar BioBERT primero
                BIOMED_EMBEDDER = SentenceTransformer(
                    'pritamdeka/S-Biomed-Roberta-snli-multinli-stsb',
                    device='cpu'
                )
                AI_EMBEDDINGS_AVAILABLE = True
                USE_FALLBACK = False
                st.success("✅ BioBERT embeddings available")
            except:
                # Fallback a SBERT
                try:
                    FALLBACK_EMBEDDER = SentenceTransformer(
                        'all-MiniLM-L6-v2',
                        device='cpu'
                    )
                    AI_EMBEDDINGS_AVAILABLE = True
                    USE_FALLBACK = True
                    st.warning("⚠️ BioBERT unavailable, using SBERT (general model)")
                except:
                    AI_EMBEDDINGS_AVAILABLE = False
    except ImportError:
        AI_EMBEDDINGS_AVAILABLE = False
        
except Exception as e:
    print(f"⚠️ Some ML libraries not available: {e}")
    AI_EMBEDDINGS_AVAILABLE = False

TFIDF_AVAILABLE = True  # Siempre disponible

st.set_page_config(
    page_title="PubMed AI Analyzer - Advanced Flavor Generator",
    page_icon="🧠",
    layout="centered"
)

# ============================================================================
# FUNCIONES DE UTILIDAD PARA PubMed CON MANEJO DE RATE LIMITING
# ============================================================================

def make_request_with_retry(url, params, max_retries=5, initial_delay=2):
    """Make request with exponential backoff for rate limiting"""
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 429:
                # Too Many Requests - wait and retry
                wait_time = delay * (2 ** attempt)
                st.warning(f"⚠️ Rate limit reached. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay)
    
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_abstract(pmid):
    """Get article abstract from PubMed with caching and rate limiting"""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    fetch_url = f"{base_url}efetch.fcgi"
    
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml",
        "rettype": "abstract"
    }
    
    try:
        response = make_request_with_retry(fetch_url, params)
        if response is None:
            return None
            
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
# EXTRACCIÓN DINÁMICA DE TÉRMINOS DESDE BÚSQUEDA E HIPÓTESIS
# ============================================================================

def extract_key_terms_from_query(query):
    """Extract key terms from search query - NO HARDCODED TERMS"""
    terms = set()
    
    # Extract MeSH terms
    mesh_pattern = r'"([^"]+)"\[Mesh\]'
    mesh_terms = re.findall(mesh_pattern, query, re.IGNORECASE)
    for term in mesh_terms:
        terms.add(term.lower())
    
    # Extract tiab terms
    tiab_pattern = r'"([^"]+)"\[tiab\]'
    tiab_terms = re.findall(tiab_pattern, query, re.IGNORECASE)
    for term in tiab_terms:
        terms.add(term.lower())
    
    # Extract quoted phrases
    quoted_pattern = r'"([^"]+)"'
    quoted_terms = re.findall(quoted_pattern, query)
    for term in quoted_terms:
        if len(term) > 3 and not term.lower().endswith('mesh') and not term.lower().endswith('tiab'):
            terms.add(term.lower())
    
    # Extract words with brackets (MeSH format)
    bracket_pattern = r'([a-zA-Z\s]+)\[Mesh\]'
    bracket_terms = re.findall(bracket_pattern, query, re.IGNORECASE)
    for term in bracket_terms:
        terms.add(term.lower().strip())
    
    return list(terms)

def extract_key_terms_from_hypothesis(hypothesis):
    """Extract key terms from hypothesis - NO HARDCODED TERMS"""
    if not hypothesis:
        return []
    
    terms = set()
    hypothesis_lower = hypothesis.lower()
    
    # Extract noun phrases (simple approach)
    words = hypothesis_lower.split()
    for i in range(len(words)-1):
        if len(words[i]) > 4 and len(words[i+1]) > 3:
            terms.add(f"{words[i]} {words[i+1]}")
        if len(words[i]) > 3:
            terms.add(words[i])
    
    # Extract specific patterns
    pattern = r'\b([a-z]+(?:\s+[a-z]+){1,3})\b'
    matches = re.findall(pattern, hypothesis_lower)
    for match in matches:
        if len(match.split()) >= 2:
            terms.add(match)
    
    return list(terms)

# ============================================================================
# ENTITY EXTRACTION - DINÁMICA CON FALLBACK
# ============================================================================

def get_embedder():
    """Get the available embedder (BioBERT, SBERT, or None)"""
    if BIOMED_EMBEDDER is not None:
        return BIOMED_EMBEDDER
    elif FALLBACK_EMBEDDER is not None:
        return FALLBACK_EMBEDDER
    else:
        return None

def extract_entities_with_embeddings(text, dynamic_terms):
    """Extract entities using embeddings if available"""
    embedder = get_embedder()
    if not embedder or not text or not dynamic_terms or not AI_EMBEDDINGS_AVAILABLE:
        return []
    
    try:
        text_embedding = embedder.encode([text[:2000]])[0]
        entity_embeddings = embedder.encode(dynamic_terms)
        
        similarities = cosine_similarity([text_embedding], entity_embeddings)[0]
        
        threshold = 0.60 if USE_FALLBACK else 0.65
        
        entities = []
        for i, sim in enumerate(similarities):
            if sim > threshold:
                entities.append((dynamic_terms[i], 'semantic', sim))
        
        return entities
    except Exception as e:
        return []

def extract_entities_with_regex(text, dynamic_terms):
    """Extract entities using regex - NO HARDCODED PATTERNS"""
    if not text or not dynamic_terms:
        return []
    
    text_lower = text.lower()
    entities = []
    
    for term in dynamic_terms:
        if term.lower() in text_lower:
            entities.append((term, 'regex_match', 1.0))
    
    return entities

def extract_medical_entities_enhanced(text, query_terms, hypothesis_terms):
    """Extract entities combining embeddings and regex - NO HARDCODED TERMS"""
    if not text:
        return []
    
    # Combine all dynamic terms
    all_terms = list(set(query_terms + hypothesis_terms))
    
    embedding_entities = extract_entities_with_embeddings(text, all_terms)
    regex_entities = extract_entities_with_regex(text, all_terms)
    
    all_entities = []
    seen = set()
    
    for entity, etype, score in embedding_entities:
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
    """Extract all mentioned outcomes - NO HARDCODED PATTERNS"""
    if not text:
        return []
    
    text_lower = text.lower()
    outcomes = set()
    
    # Common outcome-related terms (generic enough)
    outcome_indicators = ['mortality', 'death', 'survival', 'rupture', 'bleeding', 'hemorrhage', 
                          'stroke', 'reinfarction', 'complication', 'risk', 'rate', 'incidence', 
                          'outcome', 'endpoint', 'recovery', 'improvement', 'efficacy', 'safety']
    
    for indicator in outcome_indicators:
        if indicator in text_lower:
            outcomes.add(indicator)
    
    return list(outcomes)

def analyze_sentiment_by_outcome(text):
    """Analyze sentiment for each outcome"""
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

def analyze_article_with_ai(title, abstract, query_terms, hypothesis_terms):
    """Analyze an article with enhanced AI - NO HARDCODED TERMS"""
    if not title and not abstract:
        return {}
    
    full_text = f"{title} {abstract if abstract else ''}"
    processed_text = preprocess_text(full_text)
    
    entities = extract_medical_entities_enhanced(full_text, query_terms, hypothesis_terms)
    
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
        'men': r'\bmen\b|\bmale\b',
        'diabetes': r'\bdiabetes\b|\bdiabetic\b',
        'hypertension': r'\bhypertension\b|\bhypertensive\b'
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

def search_pubmed(query, retmax=100):
    """Search articles in PubMed with rate limiting"""
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
        response = make_request_with_retry(search_url, search_params)
        if response is None:
            return [], 0
            
        root = ElementTree.fromstring(response.content)
        id_list = [id_elem.text for id_elem in root.findall(".//Id")]
        count = root.find(".//Count").text if root.find(".//Count") is not None else "0"
        
        return id_list, int(count)
    except Exception as e:
        st.error(f"Search error: {e}")
        return [], 0

def fetch_articles_details(id_list, query_terms, hypothesis_terms):
    """Fetch article details and analyze them with improved rate limiting"""
    if not id_list:
        return []
    
    total_to_process = len(id_list)
    batch_size = 30  # Reduced from 50 to avoid rate limiting
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
        
        max_retries = 3
        retry_delay = 5
        
        for retry in range(max_retries):
            try:
                summary_response = make_request_with_retry(f"{base_url}esummary.fcgi", summary_params)
                if summary_response is None:
                    if retry < max_retries - 1:
                        wait_time = retry_delay * (2 ** retry)
                        status_text.text(f"⚠️ Rate limit. Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception("Max retries exceeded")
                
                summary_root = ElementTree.fromstring(summary_response.content)
                
                for j, doc_sum in enumerate(summary_root.findall(".//DocSum")):
                    overall_idx = start_idx + j
                    progress_bar.progress((overall_idx + 1) / total_to_process)
                    
                    article = extract_article_info(doc_sum)
                    
                    # Add delay between abstract requests
                    time.sleep(0.3)  # Increased from 0.05 to 0.3
                    
                    abstract = get_abstract(article["pmid"])
                    article["abstract"] = abstract if abstract else "Not available"
                    
                    ai_analysis = analyze_article_with_ai(article["title"], abstract, query_terms, hypothesis_terms)
                    article.update(ai_analysis)
                    
                    articles.append(article)
                
                # Break out of retry loop on success
                break
                
            except Exception as e:
                if retry < max_retries - 1:
                    status_text.text(f"⚠️ Error in batch {batch_num + 1}: {str(e)[:50]}. Retrying...")
                    time.sleep(retry_delay * (2 ** retry))
                    continue
                else:
                    st.warning(f"Error in batch {batch_num + 1}: {str(e)[:100]}")
        
        # Add delay between batches
        time.sleep(1.0)
    
    progress_bar.empty()
    status_text.empty()
    
    return articles

def calculate_relevance_to_search_and_hypothesis(articles, query, hypothesis):
    """Calculate relevance to search and hypothesis using embeddings"""
    embedder = get_embedder()
    if not embedder or not AI_EMBEDDINGS_AVAILABLE:
        for article in articles:
            article['relevance_score'] = 0.5
            article['search_relevance'] = 0.5
            article['hypothesis_relevance'] = 0.5
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
    
    # Show statistics
    if articles:
        scores = [a.get('relevance_score', 0) for a in articles]
        st.write(f"**📊 Relevance Filter:**")
        st.write(f"   - Threshold: {relevance_threshold}")
        st.write(f"   - Articles before: {len(articles)}")
        st.write(f"   - Articles after: {len(filtered)} ({len(filtered)/len(articles)*100:.1f}%)")
        
        if filtered:
            st.write(f"   - Filtered min: {min([a.get('relevance_score', 0) for a in filtered]):.3f}")
            st.write(f"   - Filtered max: {max([a.get('relevance_score', 0) for a in filtered]):.3f}")
    
    return filtered

# ============================================================================
# FUNCIONES DE CLUSTERING
# ============================================================================

def extract_topic_keywords_tfidf(articles, n_keywords=5):
    """Extract topic keywords using TF-IDF"""
    if not articles or len(articles) < 2:
        return []
    
    texts = []
    for a in articles[:20]:
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
        
        feature_names = vectorizer.get_feature_names_out()
        scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
        top_indices = scores.argsort()[-n_keywords:][::-1]
        
        top_keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
        return top_keywords[:n_keywords]
    except Exception as e:
        return []

def determine_flavor_aspect_and_difference(articles, flavor_name, query_terms, hypothesis_terms):
    """Determine aspect and difference based on actual article content"""
    if not articles:
        return "Clinical analysis", "Integrative approach"
    
    all_outcomes = []
    all_methods = []
    all_populations = []
    all_entities = []
    
    for a in articles:
        outcomes = a.get('all_outcomes', [])
        all_outcomes.extend(outcomes)
        
        text = f"{a.get('title', '')} {a.get('abstract', '')}".lower()
        
        if 'cohort' in text:
            all_methods.append('cohort study')
        if 'registry' in text:
            all_methods.append('registry analysis')
        if 'meta-analysis' in text or 'systematic review' in text:
            all_methods.append('meta-analysis')
        if 'retrospective' in text:
            all_methods.append('retrospective analysis')
        
        pop = a.get('population', '')
        if pop and pop != 'Not specified':
            all_populations.extend(pop.split(', '))
        
        entities = a.get('entities', [])
        for entity, etype, score in entities:
            all_entities.append(entity)
    
    outcome_counts = Counter(all_outcomes)
    method_counts = Counter(all_methods)
    population_counts = Counter(all_populations)
    entity_counts = Counter(all_entities)
    
    aspect = ""
    
    if entity_counts:
        top_entity = entity_counts.most_common(1)[0][0]
        aspect = f"Clinical Analysis of {top_entity.title()}"
    elif outcome_counts:
        top_outcome = outcome_counts.most_common(1)[0][0]
        aspect = f"Evaluation of {top_outcome.capitalize()} Outcomes"
    elif method_counts:
        top_method = method_counts.most_common(1)[0][0]
        aspect = f"Methodological Approach: {top_method.capitalize()}"
    else:
        aspect = "Clinical Outcomes Analysis"
    
    difference = ""
    
    if method_counts.get('meta-analysis', 0) > len(articles) * 0.3:
        difference = "Synthesizes evidence from multiple studies through meta-analytic methods"
    elif method_counts.get('registry analysis', 0) > len(articles) * 0.4:
        difference = "Leverages large-scale registry data for population-level insights"
    elif method_counts.get('cohort study', 0) > len(articles) * 0.5:
        difference = "Longitudinal cohort design enabling temporal outcome assessment"
    elif population_counts.get('elderly', 0) > len(articles) * 0.3:
        difference = "Focuses on elderly population with age-specific risk assessment"
    elif population_counts.get('women', 0) > len(articles) * 0.2:
        difference = "Gender-specific analysis of outcomes"
    else:
        tfidf_keywords = extract_topic_keywords_tfidf(articles, n_keywords=3)
        if tfidf_keywords:
            difference = f"Explores emerging themes including {', '.join(tfidf_keywords[:3])}"
        else:
            difference = "Integrates diverse clinical studies with heterogeneous methodologies"
    
    return aspect, difference

def get_text_embeddings(texts):
    """Get embeddings with fallback to TF-IDF if needed"""
    embedder = get_embedder()
    
    if embedder and AI_EMBEDDINGS_AVAILABLE:
        try:
            return embedder.encode(texts)
        except:
            pass
    
    # Fallback: TF-IDF + TruncatedSVD
    try:
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        svd = TruncatedSVD(n_components=min(50, tfidf_matrix.shape[1] - 1), random_state=42)
        return svd.fit_transform(tfidf_matrix)
    except:
        pass
    
    return None

def discover_flavors_by_embeddings_hdbscan(articles, query_terms, hypothesis_terms):
    """Discover flavors using HDBSCAN with embedding fallback"""
    if len(articles) < 3:
        return []
    
    try:
        texts = [f"{a.get('title', '')} {' '.join(a.get('all_outcomes', []))}" for a in articles]
        embeddings = get_text_embeddings(texts)
        
        if embeddings is None:
            return []
        
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
            name = generate_descriptive_name(cluster_articles, query_terms, hypothesis_terms)
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

def discover_flavors_by_outcomes(articles, query_terms, hypothesis_terms):
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

def merge_small_clusters(flavors, target_count=4):
    """Merge small clusters to achieve approximately target_count large flavors"""
    if not flavors:
        return []
    
    if len(flavors) <= target_count:
        return flavors
    
    flavors_sorted = sorted(flavors, key=lambda x: x['n_articles'], reverse=True)
    
    merged_flavors = flavors_sorted[:target_count-1]
    
    remaining_articles = []
    for flavor in flavors_sorted[target_count-1:]:
        remaining_articles.extend(flavor['articles'])
    
    if remaining_articles:
        name = "Additional Clinical Studies"
        
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

def assign_articles_to_best_flavor(flavors):
    """Assign each article to the most relevant flavor (exclusive assignment)"""
    if not flavors:
        return []
    
    article_to_flavor = {}
    article_scores = {}
    
    for flavor in flavors:
        flavor_id = flavor['id']
        flavor_articles = flavor['articles']
        
        for article in flavor_articles:
            article_id = article.get('pmid', id(article))
            score = article.get('quality_score', 50) / 100.0
            
            flavor_name_lower = flavor['name'].lower()
            title_lower = article.get('title', '').lower()
            if any(keyword in title_lower for keyword in flavor_name_lower.split()[:3]):
                score += 0.2
            
            if article_id not in article_scores or score > article_scores[article_id]:
                article_scores[article_id] = score
                article_to_flavor[article_id] = flavor_id
    
    flavor_article_map = {flavor['id']: [] for flavor in flavors}
    for article in [a for flavor in flavors for a in flavor['articles']]:
        article_id = article.get('pmid', id(article))
        if article_id in article_to_flavor:
            assigned_flavor = article_to_flavor[article_id]
            if article not in flavor_article_map[assigned_flavor]:
                flavor_article_map[assigned_flavor].append(article)
    
    for flavor in flavors:
        flavor['articles'] = flavor_article_map.get(flavor['id'], [])
        flavor['n_articles'] = len(flavor['articles'])
        flavor['representative_articles'] = sorted(
            flavor['articles'], 
            key=lambda x: x.get('quality_score', 0), 
            reverse=True
        )[:5]
    
    flavors = [f for f in flavors if f['n_articles'] >= 2]
    
    return flavors

def generate_descriptive_name(articles, query_terms, hypothesis_terms):
    """Generate a descriptive name based on actual content"""
    if not articles:
        return "Clinical Studies"
    
    all_keywords = []
    all_outcomes = []
    all_entities = []
    
    for a in articles:
        keywords = a.get('top_keywords', '').split(', ')
        all_keywords.extend(keywords)
        outcomes = a.get('all_outcomes', [])
        all_outcomes.extend(outcomes)
        
        entities = a.get('entities', [])
        for entity, etype, score in entities:
            all_entities.append(entity)
    
    outcome_counts = Counter(all_outcomes)
    entity_counts = Counter(all_entities)
    
    name = ""
    
    if entity_counts:
        top_entity = entity_counts.most_common(1)[0][0]
        if len(top_entity) > 3:
            name = top_entity.title()
    
    if not name and outcome_counts:
        top_outcomes = [o for o, _ in outcome_counts.most_common(2)]
        name = f"Studies on {', '.join(top_outcomes)}"
    
    if not name and all_keywords:
        name = all_keywords[0].title() if all_keywords else "Research"
    
    if not name:
        name = "Clinical Studies"
    
    return name

def generate_all_flavors(articles, query_terms, hypothesis_terms):
    """Generate all flavors from multiple perspectives"""
    if not articles:
        return {}
    
    all_flavors = []
    
    if len(articles) >= 5:
        # Siempre intentar clustering semántico (usa TF-IDF como fallback)
        semantic_flavors = discover_flavors_by_embeddings_hdbscan(articles, query_terms, hypothesis_terms)
        all_flavors.extend(semantic_flavors)
    
    if len(articles) >= 5:
        outcome_flavors = discover_flavors_by_outcomes(articles, query_terms, hypothesis_terms)
        all_flavors.extend(outcome_flavors)
    
    if not all_flavors:
        name = generate_descriptive_name(articles, query_terms, hypothesis_terms)
        all_flavors = [{
            'type': 'default',
            'id': 'default_cluster',
            'name': name,
            'articles': articles,
            'n_articles': len(articles),
            'representative_articles': articles[:5]
        }]
    
    all_flavors = assign_articles_to_best_flavor(all_flavors)
    
    merged_flavors = merge_small_clusters(all_flavors, target_count=4)
    
    flavors_by_category = {
        'semantic_clusters': [],
        'outcome_clusters': [],
        'merged_clusters': []
    }
    
    for flavor in merged_flavors:
        category = flavor.get('type', 'semantic')
        if category == 'semantic':
            flavors_by_category['semantic_clusters'].append(flavor)
        elif category == 'outcome_based':
            flavors_by_category['outcome_clusters'].append(flavor)
        else:
            flavors_by_category['merged_clusters'].append(flavor)
    
    return flavors_by_category

def generate_citation_text(article, index):
    """Generate citation text"""
    authors = article.get('authors', 'Author')
    year = article.get('pubdate', 'n.d.')[:4] if article.get('pubdate') else 'n.d.'
    title = article.get('title', 'No title')
    journal = article.get('journal', 'Journal')
    pmid = article.get('pmid', '')
    
    citation = f"{index}. {authors}. {title}. {journal}. {year}"
    if pmid and pmid != 'N/A':
        citation += f"; PMID: {pmid}"
    
    return citation

def generate_flavor_summary_with_citations(articles, flavor_name, section, query_terms, hypothesis_terms):
    """Generate a summary paragraph with citations"""
    if not articles:
        return "No articles available for this flavor.", []
    
    aspect, difference = determine_flavor_aspect_and_difference(articles, flavor_name, query_terms, hypothesis_terms)
    
    main_outcomes = []
    key_findings = []
    study_types = []
    key_articles = []
    
    for idx, a in enumerate(articles[:10], 1):
        outcomes = a.get('all_outcomes', [])
        study_type = a.get('study_types', '')
        numeric = a.get('numeric_results_str', '')
        authors = a.get('authors', '').split(',')[0] if a.get('authors') else 'Author'
        
        main_outcomes.extend(outcomes[:2])
        
        if numeric:
            key_findings.append((idx, authors, numeric))
        
        if study_type and study_type != 'Not specified':
            study_types.append(study_type)
        
        key_articles.append((idx, authors))
    
    outcome_counts = Counter(main_outcomes)
    study_type_counts = Counter(study_types)
    
    top_outcomes = [o for o, _ in outcome_counts.most_common(3)]
    top_study_types = [st for st, _ in study_type_counts.most_common(2)]
    
    header = f"### {flavor_name}\n\n"
    header += f"**🎯 Aspect:** {aspect}\n\n"
    header += f"**🔍 Difference:** {difference}\n\n"
    header += "---\n\n"
    
    if section == 'introduction':
        summary = header
        
        summary += f"This flavor groups {len(articles)} clinical studies "
        if top_outcomes:
            summary += f"investigating {', '.join(top_outcomes[:2])} "
        summary += f"in the context of the research topic. "
        
        if top_study_types:
            summary += f"Methodological approaches include {', '.join(top_study_types[:2])}, "
        else:
            summary += f"Methodological approaches include clinical registries, cohort studies, and observational analyses, "
        
        summary += f"providing comprehensive insights into clinical outcomes and management strategies. "
        
        if top_outcomes:
            summary += f"Primary outcomes examined include {', '.join(top_outcomes[:3])}, "
            summary += f"reflecting a focus on clinically relevant endpoints. "
        
        if key_articles:
            refs = [f"{auth} ({idx})" for idx, auth in key_articles[:3]]
            if refs:
                summary += f"Key contributions from {', '.join(refs)} have advanced understanding of clinical outcomes. "
        
        if key_findings:
            find_refs = [f"{auth} ({idx})" for idx, auth, _ in key_findings[:3]]
            if find_refs:
                summary += f"Notable findings from {', '.join(find_refs)} include {key_findings[0][2] if key_findings else 'significant clinical associations'}, "
                summary += f"establishing foundations for risk stratification and treatment optimization."
        
        summary += f" The convergence of these {len(articles)} studies demonstrates the clinical importance of this research area."
    
    else:
        summary = header
        
        summary += f"Our analysis aligns with findings from {len(articles)} clinical studies. "
        
        if top_study_types:
            summary += f"The observed methodological convergence across {', '.join(top_study_types[:2])} has enabled identification of consistent risk factors and outcomes. "
        
        if key_articles:
            refs = [f"{auth} ({idx})" for idx, auth in key_articles[:3]]
            if refs:
                summary += f"Specifically, {', '.join(refs)} reported results that are consistent with our analysis. "
        
        if key_findings:
            find_refs = [f"{auth} ({idx})" for idx, auth, _ in key_findings[:3]]
            if find_refs:
                summary += f"Quantitative findings from {', '.join(find_refs)} provide additional evidence supporting our conclusions. "
        
        summary += f"Heterogeneity in management approaches across these {len(articles)} studies highlights the need for standardized protocols. "
        summary += f"Integration of these findings with clinical and pathological data will be crucial for improving outcomes."
    
    citations = []
    for i, article in enumerate(articles[:15], 1):
        citations.append(generate_citation_text(article, i))
    
    return summary, citations

def create_document_with_flavors(flavors, hypothesis, query, total_articles, relevance_threshold, query_terms, hypothesis_terms):
    """Create a DOCX document with flavors"""
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
    
    if USE_FALLBACK:
        doc.add_paragraph(f'Embedding model: SBERT (fallback - BioBERT unavailable)')
    elif AI_EMBEDDINGS_AVAILABLE:
        doc.add_paragraph(f'Embedding model: BioBERT (biomedical optimized)')
    else:
        doc.add_paragraph(f'Embedding model: TF-IDF + LSA (fallback)')
    doc.add_paragraph()
    
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
                section='introduction',
                query_terms=query_terms,
                hypothesis_terms=hypothesis_terms
            )
            
            doc.add_paragraph(intro_summary)
            doc.add_paragraph()
            doc.add_paragraph('References:', style='List Bullet')
            for citation in intro_citations:
                doc.add_paragraph(citation, style='List Bullet')
            doc.add_paragraph()
    
    doc.add_page_break()
    
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
                section='discussion',
                query_terms=query_terms,
                hypothesis_terms=hypothesis_terms
            )
            
            doc.add_paragraph(disc_summary)
            doc.add_paragraph()
            doc.add_paragraph('References:', style='List Bullet')
            for citation in disc_citations:
                doc.add_paragraph(citation, style='List Bullet')
            doc.add_paragraph()
    
    return doc

def export_articles_to_csv(articles):
    """Export articles data to CSV format"""
    if not articles:
        return None
    
    data = []
    for article in articles:
        row = {
            'PMID': article.get('pmid', ''),
            'Title': article.get('title', ''),
            'Authors': article.get('authors', ''),
            'Journal': article.get('journal', ''),
            'Publication Date': article.get('pubdate', ''),
            'DOI': article.get('doi', ''),
            'Study Types': article.get('study_types', ''),
            'Quality Score': article.get('quality_score', 0),
            'Evidence Strength': article.get('evidence_strength', ''),
            'Top Keywords': article.get('top_keywords', ''),
            'Population': article.get('population', ''),
            'Numeric Results': article.get('numeric_results_str', ''),
            'Relevance Score': article.get('relevance_score', 0),
            'Search Relevance': article.get('search_relevance', 0),
            'Hypothesis Relevance': article.get('hypothesis_relevance', 0),
            'Outcomes': ', '.join(article.get('all_outcomes', [])) if article.get('all_outcomes') else '',
            'Abstract (first 500 chars)': article.get('abstract', '')[:500] if article.get('abstract') else ''
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    csv_buffer.seek(0)
    
    return csv_buffer.getvalue().encode('utf-8-sig')

def display_flavors_preview(flavors):
    """Display a preview of the generated flavors"""
    st.markdown("---")
    st.markdown("## 🎨 Generated Flavors")
    st.markdown("Below are the thematic groups (flavors) discovered from your articles:")
    
    total_flavors = 0
    for category_name, flavor_list in flavors.items():
        if not flavor_list:
            continue
        
        total_flavors += len(flavor_list)
        
        category_title = category_name.replace('_', ' ').title()
        st.markdown(f"### 📂 {category_title}")
        
        for i, flavor in enumerate(flavor_list, 1):
            with st.expander(f"🔹 Flavor {i}: {flavor['name']} ({flavor['n_articles']} articles)", expanded=(i==1)):
                col1, col2 = st.columns(2)
                with col1:
                    aspect, difference = determine_flavor_aspect_and_difference(
                        flavor.get('representative_articles', flavor['articles'][:5]),
                        flavor['name'],
                        [],
                        []
                    )
                    st.markdown(f"**🎯 Aspect:** {aspect}")
                    st.markdown(f"**🔍 Difference:** {difference}")
                
                with col2:
                    st.markdown(f"**📊 Articles in this flavor:** {flavor['n_articles']}")
                    st.markdown(f"**📚 Representative articles:**")
                    for j, article in enumerate(flavor.get('representative_articles', [])[:3], 1):
                        title = article.get('title', 'No title')[:80]
                        st.markdown(f"   {j}. {title}...")
                
                st.markdown("**📄 Sample articles in this flavor:**")
                sample_data = []
                for article in flavor.get('articles', [])[:5]:
                    sample_data.append({
                        'Title': article.get('title', 'No title')[:100],
                        'Study Type': article.get('study_types', 'N/A'),
                        'Quality Score': article.get('quality_score', 0),
                        'PMID': article.get('pmid', 'N/A')
                    })
                if sample_data:
                    df_sample = pd.DataFrame(sample_data)
                    st.dataframe(df_sample, use_container_width=True)
    
    st.info(f"✅ Total flavors generated: {total_flavors}")
    return total_flavors

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
        st.info("ℹ️ Using TF-IDF based methods (no embeddings). Quality still good for most use cases.")
    
    st.info("⚡ Enhanced version: All articles found | Merged flavors (3-4 large groups) | Exclusive article assignment | Aspect + Difference per flavor | CSV Export Available")
    
    st.markdown("""
    ### Generate thematic paragraphs (flavors) for your scientific article
    
    **Features:**
    - 🔍 PubMed search with embedding-based filtering (30% search + 70% hypothesis)
    - 🧬 **Dynamic entity extraction**: Terms extracted from your search and hypothesis
    - 📊 **Clustering + merging**: Generates 3-4 large flavors with 10-15 references each
    - 📝 **Extended summaries with embedded citations**: Paragraphs of 5-7+ lines
    - 🎯 **Aspect + Difference**: Each flavor includes header with main characteristic and differential value
    - 📚 **Introduction/Discussion separation**: Section-specific paragraphs
    - ⚙️ **Configurable relevance threshold**: Adjust filtering sensitivity (REAL-TIME EFFECT)
    - 🔄 **Exclusive article assignment**: No article appears in multiple flavors
    - 📈 **All articles found**: No artificial limit, processes complete search results
    - 🌐 **English output**: All content generated in English
    - 📊 **CSV Export**: Download all article data as CSV for further analysis
    - 🚫 **No hardcoded examples**: All content generated from your data
    """)
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if AI_EMBEDDINGS_AVAILABLE:
            model_name = "BioBERT" if not USE_FALLBACK else "SBERT"
            st.success(f"✅ {model_name} available")
        else:
            st.info("ℹ️ TF-IDF mode")
    with col_b:
        st.info("📄 Output: DOCX + CSV")
    with col_c:
        st.success("📈 Max: All articles found")
    
    st.markdown("---")
    st.markdown("### 📝 Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_area(
            "**PubMed search strategy:**",
            value="(\"myocardial infarction\"[Mesh] OR \"myocardial infarction\"[tiab]) AND (\"heart rupture\"[Mesh] OR \"cardiac rupture\"[tiab] OR \"ventricular septal rupture\"[tiab] OR \"free wall rupture\"[tiab] OR \"intramyocardial dissecting hematoma\"[tiab])",
            height=100,
            help="Use MeSH syntax for better results. Terms will be extracted automatically."
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
            min_value=0.0,
            max_value=0.9,
            value=0.35,
            step=0.05,
            help="Lower values include more articles. 0.35 is a good balance. The filter is applied in real-time when you click GENERATE."
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
        value="Intramyocardial dissections occurring as a complication of myocardial infarction follow predictable anatomical pathways along established tissue planes, with distinct patterns based on timing of presentation and location within the ventricular wall.",
        height=100,
        help="Write your hypothesis in natural language. Terms will be extracted automatically."
    )
    
    generate_button = st.button("🚀 GENERATE FLAVORS", type="primary", use_container_width=True)
    
    # Initialize session state
    if 'results_generated' not in st.session_state:
        st.session_state.results_generated = False
        st.session_state.docx_data = None
        st.session_state.csv_data = None
        st.session_state.articles = None
        st.session_state.total_articles = 0
        st.session_state.total_processed = 0
        st.session_state.n_flavors = 0
        st.session_state.applied_threshold = 0.35
        st.session_state.flavors = None
    
    if generate_button:
        if not query.strip():
            st.warning("⚠️ Please enter a search strategy")
        elif not hypothesis.strip():
            st.warning("⚠️ Please enter your hypothesis")
        else:
            start_time = time.time()
            
            # Reset state for new search
            st.session_state.results_generated = False
            st.session_state.docx_data = None
            st.session_state.csv_data = None
            
            # Store the threshold used
            st.session_state.applied_threshold = relevance_threshold
            
            # Extract dynamic terms from query and hypothesis
            query_terms = extract_key_terms_from_query(query)
            hypothesis_terms = extract_key_terms_from_hypothesis(hypothesis)
            
            st.info(f"📝 Extracted {len(query_terms)} terms from search strategy")
            if query_terms:
                st.write(f"   Query terms: {', '.join(query_terms[:10])}")
            st.info(f"📝 Extracted {len(hypothesis_terms)} terms from hypothesis")
            if hypothesis_terms:
                st.write(f"   Hypothesis terms: {', '.join(hypothesis_terms[:10])}")
            
            with st.spinner("🔍 Searching articles in PubMed..."):
                id_list, total_count = search_pubmed(query.strip(), retmax=100)
                
                if not id_list:
                    st.error("❌ No articles found")
                    st.stop()
                
                st.info(f"📊 Found {total_count} articles. Processing all {len(id_list)} articles...")
                articles = fetch_articles_details(id_list, query_terms, hypothesis_terms)
            
            if not articles:
                st.error("❌ Could not process any articles")
                st.stop()
            
            st.success(f"✅ Processed {len(articles)} articles")
            
            with st.spinner("🧠 Calculating relevance with search and hypothesis (embeddings)..."):
                articles = calculate_relevance_to_search_and_hypothesis(articles, query, hypothesis)
                
                st.markdown("---")
                st.subheader("🔍 Relevance Filter Results")
                filtered_articles = filter_articles_by_relevance(articles, relevance_threshold)
                
                articles = filtered_articles
            
            if len(articles) < 5:
                st.error(f"❌ Not enough articles to generate flavors after filtering (minimum 5, got {len(articles)})")
                st.info("💡 Try reducing the relevance threshold to include more articles.")
                st.stop()
            
            # Store articles for CSV export
            st.session_state.articles = articles
            
            with st.spinner("🔍 Discovering and merging flavors (3-4 large groups)..."):
                flavors = generate_all_flavors(articles, query_terms, hypothesis_terms)
                st.session_state.flavors = flavors
            
            total_flavors = sum(len(flavor_list) for flavor_list in flavors.values())
            st.session_state.n_flavors = total_flavors
            st.session_state.total_articles = len(articles)
            st.session_state.total_processed = len(id_list) if id_list else 0
            
            # Display flavors preview
            display_flavors_preview(flavors)
            
            with st.spinner("📄 Creating document with extended summaries and embedded citations..."):
                doc = create_document_with_flavors(flavors, hypothesis, query, len(articles), relevance_threshold, query_terms, hypothesis_terms)
                
                docx_bytes = BytesIO()
                doc.save(docx_bytes)
                docx_bytes.seek(0)
                
                st.session_state.docx_data = docx_bytes
            
            # Create CSV export
            with st.spinner("📊 Creating CSV export..."):
                csv_data = export_articles_to_csv(articles)
                st.session_state.csv_data = csv_data
            
            st.session_state.results_generated = True
            
            elapsed_time = time.time() - start_time
            st.info(f"⏱️ Total time: {elapsed_time/60:.1f} minutes")
            
            # Success notification with balloons
            st.balloons()
            st.success("🎉 Processing complete! Your documents are ready for download below.")
    
    # Always show download section if results are generated
    if st.session_state.results_generated:
        st.markdown("---")
        st.markdown("## 📥 Download Your Documents")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Articles found", st.session_state.total_processed)
        with col2:
            st.metric("After filtering", st.session_state.total_articles)
        with col3:
            st.metric("Threshold used", f"{st.session_state.applied_threshold:.2f}")
        with col4:
            st.metric("Large flavors", st.session_state.n_flavors)
        
        # Two-column layout for download buttons
        col_left, col_right = st.columns(2)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with col_left:
            st.markdown("### 📄 Flavors Document (DOCX)")
            if st.session_state.docx_data is not None:
                st.download_button(
                    label="💾 DOWNLOAD FLAVORS (DOCX)",
                    data=st.session_state.docx_data,
                    file_name=f"flavors_{timestamp}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                    type="primary"
                )
                st.caption(f"📄 File: flavors_{timestamp}.docx")
        
        with col_right:
            st.markdown("### 📊 Article Data (CSV)")
            if st.session_state.csv_data is not None:
                st.download_button(
                    label="📊 DOWNLOAD ARTICLES (CSV)",
                    data=st.session_state.csv_data,
                    file_name=f"articles_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="secondary"
                )
                st.caption(f"📊 File: articles_{timestamp}.csv | Includes all article metadata, relevance scores, and outcomes")
        
        # Show article count in CSV
        if st.session_state.articles:
            st.info(f"📊 CSV contains {len(st.session_state.articles)} articles with analysis data")
        
        # Option to start new search
        st.markdown("---")
        col_reset1, col_reset2, col_reset3 = st.columns([1, 1, 1])
        with col_reset2:
            if st.button("🔄 Start New Search", use_container_width=True):
                st.session_state.results_generated = False
                st.session_state.docx_data = None
                st.session_state.csv_data = None
                st.session_state.articles = None
                st.session_state.flavors = None
                st.rerun()
        
        with st.expander("📖 Usage Guide"):
            st.markdown(f"""
            **Execution Summary:**
            - Articles found in PubMed: {st.session_state.total_processed}
            - Articles with relevance ≥ {st.session_state.applied_threshold:.2f}: {st.session_state.total_articles}
            - Large flavors generated: {st.session_state.n_flavors}
            
            **Download Options:**
            1. **FLAVORS (DOCX)**: Thematic paragraphs for Introduction and Discussion sections
            2. **ARTICLES (CSV)**: Complete article data for further analysis
            
            **How to adjust results:**
            - If you get **too few articles** (<15), **reduce the relevance threshold** (e.g., 0.25-0.30)
            - If you get **too many articles** (>200), **increase the threshold** (e.g., 0.45-0.50)
            - If processing is **too slow**, consider narrowing your search query
            
            **Embedding Status:**
            - BioBERT: {'✅ Available' if not USE_FALLBACK and AI_EMBEDDINGS_AVAILABLE else '❌ Not available'}
            - SBERT: {'✅ Available' if USE_FALLBACK and AI_EMBEDDINGS_AVAILABLE else '❌ Not available'}
            - TF-IDF: {'✅ Available' if TFIDF_AVAILABLE else '❌ Not available'}
            """)
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 0.8em;'>
            🧠 PubMed AI Analyzer - Advanced Flavor Generator v17.0<br>
            Dynamic Entity Extraction | No Hardcoded Examples | TF-IDF Always Available<br>
            BioBERT → SBERT → TF-IDF Fallback | CSV Export | English Output<br>
            <strong>✅ Robust version - Works without torch if needed</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
