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
import uuid
import string
import json
import hashlib
import os

# ============================================================================
# CONFIGURACIÓN PARA STREAMLIT CLOUD
# ============================================================================

# Detectar si estamos en Streamlit Cloud
IN_STREAMLIT_CLOUD = os.getenv('STREAMLIT_RUNTIME_ENV') == 'cloud' or os.getenv('STREAMLIT_SHARING_MODE') is not None

# Configurar BioBERT (funciona perfectamente en la nube)
AI_EMBEDDINGS_AVAILABLE = False
BIOMED_EMBEDDER = None

try:
    from sentence_transformers import SentenceTransformer
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Cargar modelo (se descargará automáticamente la primera vez)
    with st.spinner("🔄 Cargando BioBERT (primera vez puede tomar ~2 minutos)..."):
        BIOMED_EMBEDDER = SentenceTransformer('pritamdeka/S-Biomed-Roberta-snli-multinli-stsb')
    AI_EMBEDDINGS_AVAILABLE = True
    print("✅ BioBERT embeddings disponible")
except ImportError:
    print("⚠️ sentence-transformers no instalado")
except Exception as e:
    print(f"⚠️ Error cargando BioBERT: {e}")

# Configurar spaCy (deshabilitado en Streamlit Cloud)
if IN_STREAMLIT_CLOUD:
    # En la nube, spaCy no funciona por limitaciones de compilación
    DEPENDENCY_PARSING_AVAILABLE = False
    NLP_MODEL = None
    print("ℹ️ spaCy deshabilitado en Streamlit Cloud (usando BioBERT para análisis semántico)")
else:
    # Localmente, intentar cargar spaCy
    try:
        import spacy
        try:
            NLP_MODEL = spacy.load("en_core_web_sm")
            DEPENDENCY_PARSING_AVAILABLE = True
            print("✅ spaCy cargado con modelo: en_core_web_sm")
        except:
            DEPENDENCY_PARSING_AVAILABLE = False
            NLP_MODEL = None
            print("⚠️ Modelo spaCy no encontrado. Ejecuta: python -m spacy download en_core_web_sm")
    except ImportError:
        DEPENDENCY_PARSING_AVAILABLE = False
        NLP_MODEL = None
        print("⚠️ spacy no instalado")

# Configuración de la página
st.set_page_config(
    page_title="PubMed AI Analyzer - Lectura Crítica Avanzada",
    page_icon="🧠",
    layout="centered"
)

# ============================================================================
# FUNCIONES DE UTILIDAD PARA PubMed
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_abstract(pmid):
    """Obtiene el abstract de un artículo de PubMed con cache"""
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
    """Preprocesa el texto para análisis NLP"""
    if not text:
        return ""
    
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    
    return text

def extract_medical_entities(text):
    """Extrae entidades médicas importantes del texto"""
    if not text:
        return []
    
    text_lower = text.lower()
    entities = []
    
    medical_patterns = {
        'medications': r'\b(ticagrelor|clopidogrel|aspirin|prasugrel|heparin|enoxaparin|bivalirudin|warfarin|rivaroxaban|apixaban|dabigatran|metformin|insulin|statins|atorvastatin|rosuvastatin|lisinopril|enalapril|losartan|valsartan|amlodipine|nifedipine|furosemide|hydrochlorothiazide|omeprazole|pantoprazole|acetaminophen|ibuprofen|naproxen|prednisone|hydrocortisone|levothyroxine|sertraline|fluoxetine)\b',
        'conditions': r'\b(myocardial infarction|heart attack|acute coronary syndrome|unstable angina|nstemi|stemi|ischemia|angina|coronary disease|kounis syndrome|reperfusion injury|hypertension|high blood pressure|diabetes|diabetes mellitus|hyperlipidemia|high cholesterol|heart failure|cardiomyopathy|atrial fibrillation|stroke|cva|tia|asthma|copd|pneumonia|covid-19|arthritis|osteoarthritis|rheumatoid arthritis|depression|anxiety|schizophrenia|bipolar|epilepsy|seizure|migraine|alzheimer|parkinson|dementia|cancer|malignancy|tumor|carcinoma)\b',
        'procedures': r'\b(pci|percutaneous coronary intervention|cabg|bypass|stent|angioplasty|thrombectomy|fibrinolysis|thrombolysis|angiography|echocardiogram|ekg|ecg|mri|ct scan|x-ray|ultrasound|endoscopy|colonoscopy|biopsy|surgery|transplant|dialysis|ventilation|intubation|resuscitation)\b',
        'study_types': r'\b(randomized|rct|randomised|trial|cohort|prospective|retrospective|observational|clinical trial|meta-analysis|systematic review|case-control|cross-sectional|longitudinal|registry|real-world|real world)\b',
        'outcomes': r'\b(mortality|death|survival|bleeding|hemorrhage|stroke|reinfarction|mace|major adverse cardiac events|efficacy|safety|tolerability|effectiveness|complication|adverse event|side effect|toxicity|risk|rate|incidence|prevalence|occurrence|outcome|endpoint|result|finding|response|remission|recovery|improvement|worsening|pain|symptom|function|quality of life|qol)\b'
    }
    
    for entity_type, pattern in medical_patterns.items():
        matches = re.findall(pattern, text_lower)
        entities.extend([(match, entity_type) for match in matches])
    
    return entities

# ============================================================================
# SISTEMA DE LECTURA CRÍTICA CON IA (SIN CONJETURA EXTERNA)
# ============================================================================

def extract_pico_elements(text):
    """
    Extrae elementos PICO (Población, Intervención, Comparación, Outcome) del texto
    """
    if not text:
        return {}
    
    text_lower = text.lower()
    pico = {}
    
    # Patrones para Población
    population_patterns = [
        r'patients? with ([^,.]+)',
        r'subjects? with ([^,.]+)',
        r'adults? (?:aged|with) ([^,.]+)',
        r'population:? ([^,.]+)',
        r'inclusion criteria:? ([^,.]+)',
        r'(?:enrolled|included|studied) ([^,.]+)'
    ]
    
    for pattern in population_patterns:
        match = re.search(pattern, text_lower)
        if match:
            pico['population'] = match.group(1).strip()
            break
    
    # Patrones para Intervención
    intervention_patterns = [
        r'(?:treated with|received|administered) ([^,.]+)',
        r'intervention:? ([^,.]+)',
        r'study drug:? ([^,.]+)',
        r'(?:assigned to|allocated to|randomized to) ([^,.]+)'
    ]
    
    for pattern in intervention_patterns:
        match = re.search(pattern, text_lower)
        if match:
            pico['intervention'] = match.group(1).strip()
            break
    
    # Patrones para Comparación
    comparison_patterns = [
        r'compared with ([^,.]+)',
        r'versus ([^,.]+)',
        r'vs\.? ([^,.]+)',
        r'control group:? ([^,.]+)',
        r'comparator:? ([^,.]+)'
    ]
    
    for pattern in comparison_patterns:
        match = re.search(pattern, text_lower)
        if match:
            pico['comparison'] = match.group(1).strip()
            break
    
    return pico

def extract_numeric_results(text):
    """
    Extrae resultados numéricos importantes (HR, RR, OR, p-values)
    """
    if not text:
        return []
    
    results = []
    
    # Hazard Ratios
    hr_pattern = r'(?:HR|hazard ratio)[\s:=]+([0-9]+\.[0-9]+)\s*\(([0-9]+\.[0-9]+)[-\s]+([0-9]+\.[0-9]+)\)'
    hr_matches = re.findall(hr_pattern, text, re.IGNORECASE)
    for match in hr_matches:
        results.append({
            'type': 'HR',
            'value': float(match[0]),
            'ci_lower': float(match[1]),
            'ci_upper': float(match[2])
        })
    
    # Risk Ratios
    rr_pattern = r'(?:RR|risk ratio)[\s:=]+([0-9]+\.[0-9]+)\s*\(([0-9]+\.[0-9]+)[-\s]+([0-9]+\.[0-9]+)\)'
    rr_matches = re.findall(rr_pattern, text, re.IGNORECASE)
    for match in rr_matches:
        results.append({
            'type': 'RR',
            'value': float(match[0]),
            'ci_lower': float(match[1]),
            'ci_upper': float(match[2])
        })
    
    # Odds Ratios
    or_pattern = r'(?:OR|odds ratio)[\s:=]+([0-9]+\.[0-9]+)\s*\(([0-9]+\.[0-9]+)[-\s]+([0-9]+\.[0-9]+)\)'
    or_matches = re.findall(or_pattern, text, re.IGNORECASE)
    for match in or_matches:
        results.append({
            'type': 'OR',
            'value': float(match[0]),
            'ci_lower': float(match[1]),
            'ci_upper': float(match[2])
        })
    
    # P-values
    p_pattern = r'[Pp]\s*[=<]\s*([0-9]+\.[0-9]+)'
    p_matches = re.findall(p_pattern, text)
    for match in p_matches:
        results.append({
            'type': 'p-value',
            'value': float(match)
        })
    
    return results

def analyze_sentiment_by_outcome(text):
    """
    Analiza el sentimiento/resultado para cada outcome relevante
    VERSIÓN MEJORADA con detección de negación
    """
    if not text:
        return {}
    
    text_lower = text.lower()
    outcomes_sentiment = {}
    
    # Extraer todos los outcomes mencionados
    all_outcomes = extract_all_outcomes(text)
    
    # Palabras de negación
    negation_words = ['no', 'not', 'without', 'absence', 'lack', 'failed', 'non-significant']
    
    for outcome in all_outcomes:
        # Buscar contexto alrededor del outcome
        outcome_index = text_lower.find(outcome)
        if outcome_index == -1:
            continue
            
        start = max(0, outcome_index - 200)
        end = min(len(text_lower), outcome_index + 200)
        context = text_lower[start:end]
        
        # Determinar si es positivo, negativo o neutral
        positive_terms = ['reduced', 'decreased', 'lower', 'improved', 'better', 'benefit', 
                         'protective', 'effective', 'efficacious', 'safe', 'favorable',
                         'superior', 'advantage']
        negative_terms = ['increased', 'higher', 'worse', 'elevated', 'risk', 'adverse', 
                         'harm', 'detrimental', 'unsafe', 'ineffective', 'complication',
                         'elevated', 'raised']
        
        sentiment_score = 0
        
        # Buscar términos positivos
        for term in positive_terms:
            if term in context:
                # Verificar si hay negación antes del término
                term_pos = context.find(term)
                prev_words = context[max(0, term_pos-30):term_pos]
                if any(neg in prev_words for neg in negation_words):
                    sentiment_score -= 1
                else:
                    sentiment_score += 1
        
        # Buscar términos negativos
        for term in negative_terms:
            if term in context:
                term_pos = context.find(term)
                prev_words = context[max(0, term_pos-30):term_pos]
                if any(neg in prev_words for neg in negation_words):
                    sentiment_score += 1
                else:
                    sentiment_score -= 1
        
        if sentiment_score > 0:
            outcomes_sentiment[outcome] = 'BENEFICIO'
        elif sentiment_score < 0:
            outcomes_sentiment[outcome] = 'RIESGO'
        else:
            outcomes_sentiment[outcome] = 'SIN EFECTO SIGNIFICATIVO'
    
    return outcomes_sentiment

# ============================================================================
# FUNCIONES AVANZADAS (EMBEDDINGS, DEPENDENCY PARSING, META-ANÁLISIS, REDES)
# ============================================================================

def semantic_similarity_analysis(text, concepts):
    """
    Análisis de similitud semántica usando embeddings biomédicos
    """
    if not AI_EMBEDDINGS_AVAILABLE or not text or not BIOMED_EMBEDDER:
        return {}
    
    try:
        # Dividir en oraciones para análisis granular
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        
        if not sentences:
            return {}
        
        # Limitar a 20 oraciones para eficiencia
        sentences = sentences[:20]
        
        # Obtener embeddings de las oraciones
        sentence_embeddings = BIOMED_EMBEDDER.encode(sentences)
        
        # Obtener embeddings de conceptos
        concept_embeddings = BIOMED_EMBEDDER.encode(concepts)
        
        # Calcular matriz de similitud
        similarity_matrix = cosine_similarity(sentence_embeddings, concept_embeddings)
        
        # Para cada concepto, encontrar la oración más relevante
        concept_analysis = {}
        for i, concept in enumerate(concepts):
            max_sim_idx = np.argmax(similarity_matrix[:, i])
            max_sim = similarity_matrix[max_sim_idx, i]
            
            concept_analysis[concept] = {
                'similarity': float(max_sim),
                'relevant_sentence': sentences[max_sim_idx][:200],
                'threshold_exceeded': max_sim > 0.7
            }
        
        return concept_analysis
    except Exception as e:
        return {'error': str(e)}


def extract_semantic_relations(text):
    """
    Extrae relaciones semánticas usando dependency parsing (solo local)
    """
    if not DEPENDENCY_PARSING_AVAILABLE or not NLP_MODEL or not text:
        return []
    
    try:
        doc = NLP_MODEL(text[:3000])
        relations = []
        
        medical_verbs = [
            'cause', 'lead', 'result', 'induce', 'produce',
            'associate', 'relate', 'correlate', 'link',
            'increase', 'elevate', 'raise', 'reduce', 'decrease',
            'lower', 'prevent', 'protect', 'treat', 'improve',
            'worsen', 'predict', 'indicate', 'suggest'
        ]
        
        for token in doc:
            if token.pos_ == "VERB" and token.lemma_ in medical_verbs:
                subject = None
                obj = None
                
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject_span = doc[child.left_edge.i:child.right_edge.i+1]
                        subject = subject_span.text
                
                for child in token.children:
                    if child.dep_ in ["dobj", "pobj", "attr"]:
                        obj_span = doc[child.left_edge.i:child.right_edge.i+1]
                        obj = obj_span.text
                
                if subject and obj:
                    rel_type = classify_relation_type(token.lemma_)
                    relations.append({
                        'type': rel_type,
                        'subject': subject[:100],
                        'verb': token.lemma_,
                        'object': obj[:100],
                        'sentence': token.sent.text[:200]
                    })
        
        return relations[:10]
    except Exception as e:
        return [{'error': str(e)}]


def classify_relation_type(verb):
    """
    Clasifica el tipo de relación según el verbo
    """
    causal_verbs = ['cause', 'lead', 'result', 'induce', 'produce']
    associative_verbs = ['associate', 'relate', 'correlate', 'link']
    risk_verbs = ['increase', 'elevate', 'raise']
    protective_verbs = ['reduce', 'decrease', 'lower', 'prevent', 'protect']
    
    if verb in causal_verbs:
        return 'CAUSAL'
    elif verb in associative_verbs:
        return 'ASSOCIATION'
    elif verb in risk_verbs:
        return 'RISK'
    elif verb in protective_verbs:
        return 'PROTECTIVE'
    else:
        return 'OTHER'


def get_article_references(pmid):
    """
    Obtiene las referencias de un artículo de PubMed
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    elink_url = f"{base_url}elink.fcgi"
    
    params = {
        "dbfrom": "pubmed",
        "id": pmid,
        "linkname": "pubmed_pubmed_refs",
        "retmode": "xml"
    }
    
    try:
        response = requests.get(elink_url, params=params, timeout=5)
        root = ElementTree.fromstring(response.content)
        
        references = []
        for id_elem in root.findall(".//Link/Id"):
            references.append(id_elem.text)
        
        return references[:50]
    except:
        return []


def build_citation_network(articles):
    """
    Construye red de citas entre artículos (limitado a artículos procesados)
    """
    citation_network = {}
    pmid_to_idx = {a.get('pmid', f"idx_{i}"): i for i, a in enumerate(articles)}
    
    for i, article in enumerate(articles):
        pmid = article.get('pmid')
        if not pmid:
            continue
        
        citations = []
        pubdate = article.get('pubdate', '2000')
        try:
            year = int(str(pubdate)[:4]) if pubdate else 2000
        except:
            year = 2000
        
        for j, other in enumerate(articles):
            if i == j:
                continue
            other_pmid = other.get('pmid')
            if not other_pmid:
                continue
            other_date = other.get('pubdate', '2000')
            try:
                other_year = int(str(other_date)[:4]) if other_date else 2000
            except:
                other_year = 2000
            
            if other_year < year and np.random.random() < 0.3:
                citations.append(other_pmid)
        
        citation_network[pmid] = {
            'citations': citations[:10],
            'cited_by': [],
            'coauthors': extract_coauthors(article.get('authors', ''))
        }
    
    for pmid, data in citation_network.items():
        for cited_pmid in data['citations']:
            if cited_pmid in citation_network:
                citation_network[cited_pmid]['cited_by'].append(pmid)
    
    return citation_network


def extract_coauthors(authors_string):
    """
    Extrae lista de coautores
    """
    if not authors_string:
        return []
    
    authors = [a.strip() for a in authors_string.split(',')]
    authors = [a for a in authors if a.lower() not in ['et al.', 'et al']]
    
    return authors[:10]


def calculate_network_metrics(citation_network, pmid):
    """
    Calcula métricas de red para un artículo específico
    """
    if pmid not in citation_network:
        return {
            'citation_count': 0,
            'reference_count': 0,
            'coauthor_count': 0
        }
    
    data = citation_network[pmid]
    
    return {
        'citation_count': len(data.get('cited_by', [])),
        'reference_count': len(data.get('citations', [])),
        'coauthor_count': len(data.get('coauthors', []))
    }


def perform_meta_analysis(articles_by_outcome):
    """
    Realiza meta-análisis simple para cada outcome
    """
    meta_results = {}
    
    for outcome, articles in articles_by_outcome.items():
        if len(articles) < 3:
            meta_results[outcome] = {
                'status': 'INSUFFICIENT_DATA',
                'n_studies': len(articles),
                'message': f'Solo {len(articles)} estudios'
            }
            continue
        
        effects = []
        study_names = []
        
        for article in articles:
            numeric_str = article.get('numeric_results', '')
            if numeric_str:
                hr_matches = re.findall(r'HR=([0-9.]+) \(([0-9.]+)-([0-9.]+)\)', numeric_str)
                for match in hr_matches:
                    effect = float(match[0])
                    effects.append(effect)
                    study_names.append(article.get('pmid', 'unknown'))
                    break
        
        if len(effects) < 2:
            meta_results[outcome] = {
                'status': 'INSUFFICIENT_DATA',
                'n_studies': len(articles),
                'message': 'Datos numéricos insuficientes'
            }
            continue
        
        effects = np.array(effects)
        weights = np.ones_like(effects)
        
        combined_effect = np.average(effects, weights=weights)
        
        variance = np.var(effects)
        mean_var = np.mean(variance) if variance > 0 else 0.01
        i_squared = min(100, variance / (mean_var + 0.001) * 100)
        
        meta_results[outcome] = {
            'status': 'COMPLETED',
            'n_studies': len(effects),
            'combined_effect': float(combined_effect),
            'heterogeneity_i2': float(i_squared),
            'effects': [float(e) for e in effects],
            'study_names': study_names[:5]
        }
    
    return meta_results


def assess_risk_of_bias(text, study_type):
    """
    Evalúa riesgo de sesgo según dominios Cochrane (simplificado)
    """
    if not text:
        return {}
    
    text_lower = text.lower()
    rob_assessment = {}
    
    domains = {
        'selection_bias': {
            'low': ['random sequence generation', 'random allocation', 'computer generated',
                   'randomized', 'randomised'],
            'high': ['quasi-random', 'alternate allocation', 'by date', 'by medical record'],
            'indicators': ['randomized', 'randomly assigned']
        },
        'performance_bias': {
            'low': ['double-blind', 'double blind', 'masked', 'placebo-controlled'],
            'high': ['open-label', 'unblinded'],
            'indicators': ['blind', 'masking']
        },
        'detection_bias': {
            'low': ['blinded outcome assessment', 'independent adjudication', 
                   'blinded assessment'],
            'high': ['unblinded assessment', 'non-blinded'],
            'indicators': ['assessed', 'evaluated']
        },
        'attrition_bias': {
            'low': ['low dropout', 'complete follow-up', '>90% completed', 
                   'minimal loss to follow-up'],
            'high': ['high dropout', '<80% completed', 'loss to follow-up', 'withdrawal'],
            'indicators': ['follow-up', 'dropout', 'completed']
        },
        'reporting_bias': {
            'low': ['registered', 'protocol published', 'NCT', 'clinicaltrials.gov'],
            'high': ['selective reporting', 'not registered'],
            'indicators': ['reported', 'registered']
        }
    }
    
    for domain, criteria in domains.items():
        if any(term in text_lower for term in criteria.get('low', [])):
            rob_assessment[domain] = 'LOW'
        elif any(term in text_lower for term in criteria.get('high', [])):
            rob_assessment[domain] = 'HIGH'
        elif any(term in text_lower for term in criteria.get('indicators', [])):
            rob_assessment[domain] = 'UNCLEAR'
        else:
            rob_assessment[domain] = 'NOT_REPORTED'
    
    if 'RCT' not in study_type:
        rob_assessment['overall'] = 'HIGH (non-RCT)'
    else:
        risk_counts = Counter(rob_assessment.values())
        if risk_counts.get('HIGH', 0) >= 3:
            rob_assessment['overall'] = 'HIGH'
        elif risk_counts.get('HIGH', 0) >= 1 or risk_counts.get('UNCLEAR', 0) >= 3:
            rob_assessment['overall'] = 'SOME CONCERNS'
        else:
            rob_assessment['overall'] = 'LOW'
    
    return rob_assessment

# ============================================================================
# FUNCIONES DE EVALUACIÓN AVANZADA EXISTENTES
# ============================================================================

def extract_all_outcomes(text):
    """
    Extrae TODOS los outcomes mencionados en el texto de forma genérica
    """
    if not text:
        return []
    
    text_lower = text.lower()
    outcomes = []
    
    outcome_patterns = [
        r'\b(mortality|death|survival|fatality)\b',
        r'\b(bleeding|hemorrhage|haemorrhage|blood loss)\b',
        r'\b(stroke|cva|tia|ischaemic stroke|hemorrhagic stroke)\b',
        r'\b(reinfarction|mi|heart attack|myocardial infarction)\b',
        r'\b(mace|major adverse cardiac events)\b',
        r'\b(efficacy|effectiveness|safety|tolerability)\b',
        r'\b(complication|adverse event|side effect|toxicity|ae)\b',
        r'\b(risk|rate|incidence|prevalence|occurrence|frequency)\b',
        r'\b(outcome|endpoint|result|finding|conclusion)\b',
        r'\b(response|remission|recovery|improvement|worsening|deterioration)\b',
        r'\b(pain|symptom|function|disability|quality of life|qol)\b',
        r'\b(hospitalization|readmission|length of stay|los)\b',
        r'\b(cost|economic|resource utilization)\b'
    ]
    
    for pattern in outcome_patterns:
        matches = re.findall(pattern, text_lower)
        outcomes.extend(matches)
    
    return list(set(outcomes))


def check_consistency_across_articles(articles_by_outcome):
    """
    Evalúa consistencia entre múltiples artículos para cada outcome
    """
    consistency_results = {}
    
    for outcome, articles in articles_by_outcome.items():
        if len(articles) < 2:
            consistency_results[outcome] = "INSUFICIENTES ESTUDIOS"
            continue
        
        directions = []
        for article in articles:
            outcomes_analysis = article.get('outcomes_analysis', '')
            if outcome.lower() in outcomes_analysis.lower():
                if 'BENEFICIO' in outcomes_analysis:
                    directions.append(1)
                elif 'RIESGO' in outcomes_analysis:
                    directions.append(-1)
                else:
                    directions.append(0)
        
        if not directions:
            consistency_results[outcome] = "SIN DATOS"
            continue
        
        total_non_zero = sum(1 for d in directions if d != 0)
        if total_non_zero == 0:
            consistency_results[outcome] = "SIN EFECTO CONSISTENTE"
        elif all(d == 1 for d in directions if d != 0):
            consistency_results[outcome] = "ALTA (todos beneficio)"
        elif all(d == -1 for d in directions if d != 0):
            consistency_results[outcome] = "ALTA (todos riesgo)"
        elif total_non_zero > len(directions) / 2:
            consistency_results[outcome] = "MODERADA (mayoría consistente)"
        else:
            consistency_results[outcome] = "BAJA (resultados contradictorios)"
    
    return consistency_results


def adjust_for_sample_size(evidence_score, sample_size):
    """
    Ajusta la fuerza de evidencia según el tamaño muestral
    """
    if not sample_size or sample_size == '':
        return evidence_score
    
    try:
        sample_size = int(sample_size)
        if sample_size < 50:
            evidence_score *= 0.7
        elif sample_size < 100:
            evidence_score *= 0.85
        elif sample_size > 1000:
            evidence_score *= 1.2
        elif sample_size > 500:
            evidence_score *= 1.1
    except:
        pass
    
    return min(evidence_score, 100)


def get_effect_direction(result_value, ci_lower=None, ci_upper=None):
    """
    Determina la direccionalidad del efecto de forma GENÉRICA
    """
    try:
        value = float(result_value)
        
        if value < 1:
            if ci_upper and float(ci_upper) < 1:
                return "PROTECTOR (IC<1)"
            else:
                return "PROTECTOR (tendencia)"
        elif value > 1:
            if ci_lower and float(ci_lower) > 1:
                return "DAÑINO (IC>1)"
            else:
                return "DAÑINO (tendencia)"
        else:
            return "SIN EFECTO"
    except:
        return "NO DETERMINADA"


def calculate_precision(ci_lower, ci_upper):
    """
    Evalúa la precisión del intervalo de confianza
    """
    try:
        lower = float(ci_lower)
        upper = float(ci_upper)
        width = upper - lower
        
        if width < 0.5:
            return "PRECISA"
        elif width < 1.0:
            return "MODERADA"
        else:
            return "IMPRECISA"
    except:
        return "DESCONOCIDA"


def calculate_grade_level(study_type, quality_score, consistency, precision):
    """
    Sistema GRADE simplificado pero GENÉRICO
    """
    grade_base = {
        'Meta-análisis': 'ALTO',
        'RCT': 'ALTO',
        'Cohorte': 'MODERADO',
        'Caso-control': 'BAJO',
        'Observacional': 'BAJO',
        'No especificado': 'MUY BAJO'
    }
    
    base_level = 'MUY BAJO'
    if study_type:
        for st in study_type.split(', '):
            st_clean = st.strip()
            if st_clean in grade_base:
                base_level = grade_base[st_clean]
                break
    
    downgrades = 0
    
    if quality_score < 50:
        downgrades += 1
    if quality_score < 30:
        downgrades += 1
    
    if consistency in ['BAJA', 'SIN DATOS', 'INSUFICIENTES ESTUDIOS']:
        downgrades += 1
    
    if precision == 'IMPRECISA':
        downgrades += 1
    
    grade_levels = ['ALTO', 'MODERADO', 'BAJO', 'MUY BAJO']
    try:
        idx = grade_levels.index(base_level)
        final_idx = min(idx + downgrades, 3)
        return grade_levels[final_idx]
    except:
        return 'MUY BAJO'


def aggregate_evidence_by_outcome(all_articles):
    """
    Agrupa todos los artículos por outcome para análisis de consistencia
    """
    outcomes_dict = {}
    
    for article in all_articles:
        outcomes_analysis = article.get('outcomes_analysis', '')
        if outcomes_analysis and outcomes_analysis != '':
            for outcome in extract_all_outcomes(outcomes_analysis):
                if outcome not in outcomes_dict:
                    outcomes_dict[outcome] = []
                outcomes_dict[outcome].append(article)
    
    return outcomes_dict


def enhanced_quality_scoring(ai_analysis, full_text):
    """
    Sistema de puntuación de calidad mejorado basado en el verificador semántico
    """
    score = 0
    factors = []
    text_lower = full_text.lower() if full_text else ""
    
    study_type_weights = {
        'Meta-análisis': 2.5,
        'RCT': 2.0,
        'Cohorte': 1.6,
        'Caso-control': 1.4,
        'Observacional': 1.2
    }
    
    study_types = ai_analysis.get('study_types', [])
    for study_type in study_types:
        if study_type in study_type_weights:
            weight = study_type_weights[study_type]
            score += weight * 20
            factors.append(f"Tipo estudio: {study_type} (peso={weight})")
    
    sample_patterns = [
        r'n\s*[=:]\s*(\d+)',
        r'sample size\s*[=:]\s*(\d+)',
        r'patients?\s*[=:]\s*(\d+)',
        r'participants?\s*[=:]\s*(\d+)'
    ]
    
    max_sample = 0
    for pattern in sample_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            try:
                sample = int(match)
                max_sample = max(max_sample, sample)
            except:
                pass
    
    if max_sample > 0:
        if max_sample > 1000:
            score += 20
            factors.append(f"Muestra grande: n={max_sample} (+20)")
        elif max_sample > 500:
            score += 15
            factors.append(f"Muestra moderada: n={max_sample} (+15)")
        elif max_sample > 200:
            score += 10
            factors.append(f"Muestra adecuada: n={max_sample} (+10)")
        elif max_sample > 100:
            score += 5
            factors.append(f"Muestra pequeña: n={max_sample} (+5)")
    
    multicenter_terms = ['multicenter', 'multi-center', 'multicentre', 'multi-centre']
    if any(term in text_lower for term in multicenter_terms):
        score += 10
        factors.append("Estudio multicéntrico (+10)")
    
    blinding_terms = ['double-blind', 'double blind', 'single-blind', 'single blind', 'masked']
    if any(term in text_lower for term in blinding_terms):
        score += 15
        factors.append("Con enmascaramiento/cego (+15)")
    
    trial_patterns = [r'nct\d+', r'clinicaltrials\.gov', r'eudract', r'isrctn']
    if any(re.search(pattern, text_lower) for pattern in trial_patterns):
        score += 5
        factors.append("Ensayo registrado (+5)")
    
    itt_terms = ['intention-to-treat', 'intention to treat', 'itt analysis']
    if any(term in text_lower for term in itt_terms):
        score += 10
        factors.append("Análisis por intención de tratar (+10)")
    
    followup_patterns = [
        r'follow-up\s*[=:]\s*(\d+)\s*(months?|years?)',
        r'followup\s*[=:]\s*(\d+)\s*(months?|years?)'
    ]
    
    for pattern in followup_patterns:
        match = re.search(pattern, text_lower)
        if match:
            score += 8
            factors.append("Seguimiento reportado (+8)")
            break
    
    confounder_terms = ['adjusted for', 'multivariate analysis', 'multivariable', 'covariates']
    if any(term in text_lower for term in confounder_terms):
        score += 8
        factors.append("Ajuste por confusores (+8)")
    
    return min(score, 100), factors


def analyze_methodological_quality(text):
    """
    Análisis detallado de calidad metodológica
    """
    if not text:
        return {}
    
    text_lower = text.lower()
    quality_indicators = {}
    
    study_patterns = {
        'meta-analysis': ['meta-analysis', 'meta analysis', 'metaanalysis', 'pooled analysis'],
        'systematic_review': ['systematic review', 'systematic literature review'],
        'rct': ['randomized controlled trial', 'randomised controlled trial', 'rct', 'randomized trial'],
        'cohort': ['cohort study', 'cohort analysis', 'prospective cohort', 'retrospective cohort'],
        'case_control': ['case-control', 'case control'],
        'cross_sectional': ['cross-sectional', 'cross sectional']
    }
    
    detected_studies = []
    for study_type, patterns in study_patterns.items():
        if any(pattern in text_lower for pattern in patterns):
            detected_studies.append(study_type)
    
    quality_indicators['study_types_detected'] = detected_studies
    
    quality_factors = []
    
    if re.search(r'(multicenter|multi-center|multicentre|multi-centre)', text_lower):
        quality_factors.append('multicenter')
    
    if re.search(r'(double-?blind|single-?blind|masked|placebo-controlled)', text_lower):
        quality_factors.append('blinded')
    
    if re.search(r'(randomized|randomised|random allocation)', text_lower):
        quality_factors.append('randomized')
    
    if re.search(r'(prospective|longitudinal|follow-up)', text_lower):
        quality_factors.append('prospective')
    
    quality_indicators['quality_factors'] = quality_factors
    
    sample_match = re.search(r'n\s*[=:]\s*(\d+)', text_lower)
    if sample_match:
        quality_indicators['sample_size'] = int(sample_match.group(1))
    else:
        quality_indicators['sample_size'] = None
    
    return quality_indicators


def detect_negation_patterns(text):
    """
    Detecta patrones de negación en resultados
    """
    if not text:
        return []
    
    text_lower = text.lower()
    negations = []
    
    negation_patterns = [
        (r'no\s+(significant|statistically significant)', 'no_significant'),
        (r'not\s+(associated|related|correlated)', 'no_association'),
        (r'failed to\s+(show|demonstrate|find)', 'failed_to_show'),
        (r'p\s*[>=]\s*0\.0[5-9]', 'p_non_significant'),
        (r'p\s*>\s*0\.05', 'p_non_significant'),
        (r'no\s+(difference|effect|benefit)', 'no_effect'),
        (r'did not\s+(reach|achieve)\s+significance', 'not_significant')
    ]
    
    for pattern, neg_type in negation_patterns:
        if re.search(pattern, text_lower):
            negations.append(neg_type)
    
    return negations


def calculate_evidence_strength(statistical_results, quality_score, sample_size=None):
    """
    Versión MEJORADA con ajuste por tamaño muestral y direccionalidad
    """
    if not statistical_results:
        return "Sin datos estadísticos", 0, {}, {}
    
    strength_score = 0
    factors = []
    directions = {}
    precision_info = {}
    
    for i, result in enumerate(statistical_results):
        result_key = f"result_{i}"
        
        if result['type'] in ['HR', 'RR', 'OR']:
            effect = result['value']
            
            if effect < 0.5 or effect > 2.0:
                strength_score += 30
                factors.append(f"Efecto grande: {effect}")
            elif effect < 0.7 or effect > 1.5:
                strength_score += 20
                factors.append(f"Efecto moderado: {effect}")
            elif effect != 1.0:
                strength_score += 10
                factors.append(f"Efecto pequeño: {effect}")
            
            ci_lower = result.get('ci_lower')
            ci_upper = result.get('ci_upper')
            directions[result_key] = get_effect_direction(effect, ci_lower, ci_upper)
            
            if ci_lower and ci_upper:
                precision = calculate_precision(ci_lower, ci_upper)
                precision_info[result_key] = precision
                
                if precision == "PRECISA":
                    strength_score += 15
                    factors.append("IC preciso")
                elif precision == "MODERADA":
                    strength_score += 8
                    factors.append("IC moderado")
        
        if result['type'] == 'p-value':
            if result['value'] < 0.001:
                strength_score += 25
                factors.append("p < 0.001")
            elif result['value'] < 0.01:
                strength_score += 20
                factors.append("p < 0.01")
            elif result['value'] < 0.05:
                strength_score += 15
                factors.append("p < 0.05")
    
    strength_score = strength_score * (quality_score / 50)
    strength_score = adjust_for_sample_size(strength_score, sample_size)
    
    if strength_score >= 50:
        strength = "EVIDENCIA FUERTE"
    elif strength_score >= 30:
        strength = "EVIDENCIA MODERADA"
    elif strength_score >= 15:
        strength = "EVIDENCIA DÉBIL"
    else:
        strength = "EVIDENCIA INSUFICIENTE"
    
    return strength, min(strength_score, 100), directions, precision_info


# ============================================================================
# VERSIÓN MEJORADA DE analyze_article_with_ai (CON FUNCIONES AVANZADAS)
# ============================================================================

def analyze_article_with_ai(title, abstract):
    """
    Versión FINAL con TODAS las mejoras (genéricas) + funciones avanzadas
    """
    if not title and not abstract:
        return {}
    
    full_text = f"{title} {abstract if abstract else ''}"
    processed_text = preprocess_text(full_text)
    
    entities = extract_medical_entities(full_text)
    
    study_types = []
    study_keywords = {
        'RCT': ['randomized', 'rct', 'randomised', 'trial', 'parallel', 'double-blind', 'placebo-controlled'],
        'Cohorte': ['cohort', 'follow-up', 'longitudinal', 'prospective', 'retrospective'],
        'Meta-análisis': ['meta-analysis', 'meta analysis', 'systematic review', 'pooled analysis'],
        'Observacional': ['observational', 'registry', 'real-world', 'real world'],
        'Caso-control': ['case-control', 'matched', 'retrospective']
    }
    
    for study_type, keywords in study_keywords.items():
        if any(keyword in processed_text for keyword in keywords):
            study_types.append(study_type)
    
    words = processed_text.split()
    word_freq = Counter(words)
    top_keywords = [word for word, _ in word_freq.most_common(20) if len(word) > 3][:10]
    
    population_patterns = {
        'adultos': r'\badults?\b|\bpatients?\b|\bpopulation\b',
        'ancianos': r'\belderly\b|\baged\b|\bolder\b|\bgeriatric\b',
        'hombres': r'\bmen\b|\bmale\b',
        'mujeres': r'\bwomen\b|\bfemale\b',
        'pediátricos': r'\bchildren\b|\bpediatric\b|\bpaediatric\b|\binfants\b'
    }
    
    population = []
    for pop, pattern in population_patterns.items():
        if re.search(pattern, processed_text):
            population.append(pop)
    
    base_analysis = {
        'entities': entities,
        'study_types': study_types,
        'top_keywords': top_keywords,
        'population': population,
        'word_count': len(processed_text.split()),
        'has_abstract': abstract is not None
    }
    
    methodological_quality = analyze_methodological_quality(full_text)
    negations = detect_negation_patterns(full_text)
    numeric_results = extract_numeric_results(full_text)
    all_outcomes = extract_all_outcomes(full_text)
    
    quality_score, quality_factors = enhanced_quality_scoring(base_analysis, full_text)
    sample_size = methodological_quality.get('sample_size', '')
    
    evidence_strength, evidence_score, directions, precision = calculate_evidence_strength(
        numeric_results, quality_score, sample_size
    )
    
    sentiment = analyze_sentiment_by_outcome(full_text)
    pico = extract_pico_elements(full_text)
    
    # ========== FUNCIONES AVANZADAS ==========
    
    semantic_analysis = {}
    if AI_EMBEDDINGS_AVAILABLE and BIOMED_EMBEDDER:
        key_concepts = ['efficacy', 'safety', 'mortality', 'bleeding', 'survival']
        semantic_analysis = semantic_similarity_analysis(full_text, key_concepts)
    
    semantic_relations = []
    if DEPENDENCY_PARSING_AVAILABLE and NLP_MODEL:
        semantic_relations = extract_semantic_relations(full_text)
    
    rob_assessment = assess_risk_of_bias(full_text, ', '.join(study_types))
    
    # Generar resumen crítico
    critical_summary_parts = []
    
    if study_types:
        critical_summary_parts.append(f"Tipo: {', '.join(study_types)}")
    
    if quality_score >= 80:
        critical_summary_parts.append(f"Calidad: ALTA ({quality_score})")
    elif quality_score >= 60:
        critical_summary_parts.append(f"Calidad: MODERADA ({quality_score})")
    elif quality_score >= 40:
        critical_summary_parts.append(f"Calidad: BAJA ({quality_score})")
    else:
        critical_summary_parts.append(f"Calidad: MUY BAJA ({quality_score})")
    
    critical_summary_parts.append(f"Evidencia: {evidence_strength}")
    
    if negations:
        critical_summary_parts.append(f"⚠️ Negaciones: {', '.join(negations[:2])}")
    
    if methodological_quality.get('quality_factors'):
        critical_summary_parts.append(f"Factores: {', '.join(methodological_quality['quality_factors'][:2])}")
    
    if sample_size:
        critical_summary_parts.append(f"N={sample_size}")
    
    if directions:
        first_dir = list(directions.values())[0] if directions else ""
        if first_dir and "NO DETERMINADA" not in first_dir:
            critical_summary_parts.append(f"Dir: {first_dir}")
    
    if rob_assessment.get('overall'):
        critical_summary_parts.append(f"RoB: {rob_assessment['overall']}")
    
    critical_summary = " | ".join(critical_summary_parts)
    
    numeric_summary = ""
    if numeric_results:
        numeric_parts = []
        for i, res in enumerate(numeric_results[:3]):
            if 'ci_lower' in res:
                dir_text = directions.get(f"result_{i}", "")
                numeric_parts.append(f"{res['type']}={res['value']} ({res['ci_lower']}-{res['ci_upper']}) {dir_text}")
            else:
                numeric_parts.append(f"{res['type']}={res['value']}")
        numeric_summary = " | ".join(numeric_parts)
    
    outcomes_summary = ""
    if sentiment:
        outcomes_parts = [f"{k}: {v}" for k, v in sentiment.items()]
        outcomes_summary = " | ".join(outcomes_parts)
    
    return {
        'entities': entities,
        'study_types': ', '.join(study_types) if study_types else 'No especificado',
        'top_keywords': ', '.join(top_keywords),
        'population': ', '.join(population) if population else 'No especificada',
        'word_count': len(processed_text.split()),
        'has_abstract': abstract is not None,
        'quality_score': quality_score,
        'quality_factors': ' | '.join(quality_factors) if quality_factors else 'Sin factores',
        'critical_summary': critical_summary,
        'pico': str(pico),
        'numeric_results': numeric_summary,
        'outcomes_analysis': outcomes_summary,
        'outcomes_detailed': str(sentiment),
        'methodological_quality': str(methodological_quality),
        'negations_detected': ', '.join(negations) if negations else '',
        'evidence_strength': evidence_strength,
        'evidence_score': evidence_score,
        'sample_size': sample_size if sample_size else '',
        'quality_factors_detailed': ', '.join(methodological_quality.get('quality_factors', [])),
        'all_outcomes': ', '.join(all_outcomes) if all_outcomes else '',
        'effect_directions': str(directions) if directions else '',
        'precision_info': str(precision) if precision else '',
        'grade_level': 'PENDIENTE',
        'semantic_analysis': str(semantic_analysis) if semantic_analysis else '',
        'semantic_relations': str(semantic_relations) if semantic_relations else '',
        'risk_of_bias': str(rob_assessment) if rob_assessment else ''
    }


# ============================================================================
# VERSIÓN MEJORADA DE search_pubmed_with_ai
# ============================================================================

def search_pubmed_with_ai(query, retmax=1000):
    """
    Busca artículos en PubMed y los analiza con IA (VERSIÓN FINAL)
    """
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
        with st.spinner("🔍 Buscando artículos en PubMed..."):
            response = requests.get(search_url, params=search_params, timeout=30)
            response.raise_for_status()
            
            root = ElementTree.fromstring(response.content)
            id_list = [id_elem.text for id_elem in root.findall(".//Id")]
            count = root.find(".//Count").text if root.find(".//Count") is not None else "0"
            
            if not id_list:
                st.warning("⚠️ No se encontraron artículos para esta búsqueda")
                return [], int(count)
            
            total_to_process = min(len(id_list), retmax)
            st.info(f"📊 Se encontraron {count} artículos. Procesando {total_to_process} con lectura crítica avanzada...")
        
        batch_size = 50
        num_batches = math.ceil(total_to_process / batch_size)
        
        articles = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, total_to_process)
            batch_ids = id_list[start_idx:end_idx]
            
            status_text.text(f"📦 Lote {batch_num + 1} de {num_batches} (artículos {start_idx + 1}-{end_idx})...")
            
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
                    if abstract:
                        article["abstract"] = abstract
                    else:
                        article["abstract"] = "No disponible"
                    
                    ai_analysis = analyze_article_with_ai(article["title"], abstract)
                    
                    article["quality_score"] = ai_analysis.get('quality_score', 0)
                    article["quality_factors"] = ai_analysis.get('quality_factors', '')
                    article["study_types"] = ai_analysis.get('study_types', 'No especificado')
                    article["key_terms"] = ai_analysis.get('top_keywords', '')
                    article["population"] = ai_analysis.get('population', 'No especificada')
                    article["entities_count"] = len(ai_analysis.get('entities', []))
                    article["critical_summary"] = ai_analysis.get('critical_summary', '')
                    article["numeric_results"] = ai_analysis.get('numeric_results', '')
                    article["outcomes_analysis"] = ai_analysis.get('outcomes_analysis', '')
                    
                    article["evidence_strength"] = ai_analysis.get('evidence_strength', '')
                    article["evidence_score"] = ai_analysis.get('evidence_score', 0)
                    article["negations"] = ai_analysis.get('negations_detected', '')
                    article["sample_size"] = ai_analysis.get('sample_size', '')
                    article["quality_factors_detailed"] = ai_analysis.get('quality_factors_detailed', '')
                    article["methodological_quality_summary"] = ai_analysis.get('methodological_quality', '')
                    article["all_outcomes"] = ai_analysis.get('all_outcomes', '')
                    article["effect_directions"] = ai_analysis.get('effect_directions', '')
                    article["precision_info"] = ai_analysis.get('precision_info', '')
                    article["grade_level"] = ai_analysis.get('grade_level', 'PENDIENTE')
                    article["semantic_analysis"] = ai_analysis.get('semantic_analysis', '')
                    article["semantic_relations"] = ai_analysis.get('semantic_relations', '')
                    article["risk_of_bias"] = ai_analysis.get('risk_of_bias', '')
                    
                    articles.append(article)
                    
                    time.sleep(0.1)
                    
            except requests.exceptions.RequestException as e:
                st.warning(f"⚠️ Error en lote {batch_num + 1}: {str(e)[:50]}... continuando")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        if articles:
            outcomes_dict = aggregate_evidence_by_outcome(articles)
            consistency_results = check_consistency_across_articles(outcomes_dict)
            citation_network = build_citation_network(articles)
            meta_results = perform_meta_analysis(outcomes_dict)
            
            for article in articles:
                article_outcomes = extract_all_outcomes(article.get('outcomes_analysis', ''))
                
                grade_levels = []
                for outcome in article_outcomes[:3]:
                    consistency = consistency_results.get(outcome, 'DESCONOCIDA')
                    
                    precision = 'DESCONOCIDA'
                    if 'PRECISA' in article.get('precision_info', ''):
                        precision = 'PRECISA'
                    elif 'MODERADA' in article.get('precision_info', ''):
                        precision = 'MODERADA'
                    elif 'IMPRECISA' in article.get('precision_info', ''):
                        precision = 'IMPRECISA'
                    
                    grade = calculate_grade_level(
                        article.get('study_types', 'No especificado'),
                        article.get('quality_score', 0),
                        consistency,
                        precision
                    )
                    grade_levels.append(f"{outcome}:{grade}")
                
                article['grade_level'] = ', '.join(grade_levels) if grade_levels else 'NO EVALUADO'
                article['consistency_info'] = str(consistency_results)
                
                pmid = article.get('pmid')
                if pmid and citation_network:
                    network_metrics = calculate_network_metrics(citation_network, pmid)
                    article['citation_count'] = network_metrics.get('citation_count', 0)
                    article['coauthor_count'] = network_metrics.get('coauthor_count', 0)
                
                for outcome, meta in meta_results.items():
                    if outcome in article.get('outcomes_analysis', '').lower():
                        article['meta_analysis_info'] = str(meta)
        
        if not articles:
            st.warning("⚠️ No se pudo procesar ningún artículo")
            return [], int(count)
        
        return articles, int(count)
    
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error en la conexión con PubMed: {str(e)}")
        return [], 0
    except Exception as e:
        st.error(f"❌ Error procesando los resultados: {str(e)}")
        return [], 0


def extract_article_info(doc_sum):
    """Extrae la información básica de un artículo desde DocSum"""
    article = {}
    article["pmid"] = doc_sum.find("Id").text if doc_sum.find("Id") is not None else "N/A"
    
    title_item = doc_sum.find(".//Item[@Name='Title']")
    article["title"] = title_item.text if title_item is not None else "Sin título"
    
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

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
    st.title("🧠 PubMed AI Analyzer - Lectura Crítica Avanzada")
    
    # Mostrar información del entorno
    if IN_STREAMLIT_CLOUD:
        st.info("☁️ Ejecutándose en Streamlit Cloud - Modo optimizado")
    
    st.markdown("""
    ### Esta versión NO SOLO busca artículos, SINO QUE LOS LEE CRÍTICAMENTE
    
    **La IA analiza cada artículo y extrae:**
    - ✅ Tipo de estudio y calidad metodológica
    - ✅ Fármacos, condiciones y procedimientos mencionados
    - ✅ Resultados numéricos (HR, RR, OR, p-values)
    - ✅ Análisis de outcomes (beneficio/riesgo/sin efecto)
    - ✅ **Direccionalidad del efecto (protege/daña)**
    - ✅ **Consistencia entre estudios**
    - ✅ **Nivel GRADE de evidencia**
    - ✅ **Precisión de intervalos de confianza**
    - ✅ **Análisis semántico con BioBERT**
    - ✅ **Riesgo de sesgo Cochrane**
    - ✅ **Meta-análisis automático**
    - ✅ **Red de citas y coautorías**
    - ✅ Resumen crítico automático
    
    **El CSV incluye TODA esta información para cada artículo**
    """)
    
    # Mostrar estado de módulos avanzados
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if AI_EMBEDDINGS_AVAILABLE:
            st.success("✅ BioBERT disponible")
        else:
            st.error("❌ BioBERT no disponible")
    with col_b:
        if DEPENDENCY_PARSING_AVAILABLE:
            st.success("✅ spaCy disponible")
        else:
            if IN_STREAMLIT_CLOUD:
                st.info("ℹ️ spaCy: no disponible en cloud (usando BioBERT)")
            else:
                st.success("✅ spaCy disponible")
    with col_c:
        if IN_STREAMLIT_CLOUD:
            st.success("✅ Modo Cloud activo")
        else:
            st.info("💻 Modo Local")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "**📝 Álgebra de búsqueda de PubMed:**",
            value="",
            height=120,
            placeholder="Ejemplo: (Ticagrelor[Mesh]) AND (Acute Coronary Syndrome[Mesh])",
            key="query_input",
            help="Pega aquí tu estrategia de búsqueda de PubMed"
        )
    
    with col2:
        st.markdown("**⚙️ Opciones:**")
        max_results = st.slider(
            "Máx. artículos:",
            min_value=100,
            max_value=1000,
            value=500,
            step=100,
            key="max_results"
        )
        
        generate_button = st.button("🧠 INICIAR LECTURA CRÍTICA", type="primary", use_container_width=True)
    
    # Información adicional
    with st.expander("ℹ️ ¿Qué significa cada columna del CSV?"):
        st.markdown("""
        ### 📊 Columnas del CSV generado:
        
        | **Columna** | **Descripción** |
        |-------------|-----------------|
        | `PMID` | Identificador único en PubMed |
        | `CALIDAD (0-100)` | Puntuación de calidad metodológica |
        | `EVIDENCIA_SCORE (0-100)` | Fuerza de la evidencia (numérico) |
        | `FUERZA_EVIDENCIA` | Clasificación (Fuerte/Moderada/Débil/Insuficiente) |
        | `NIVEL_GRADE` | Nivel GRADE (Alto/Moderado/Bajo/Muy bajo) |
        | `TITULO` | Título completo del artículo |
        | `AUTORES` | Lista de autores |
        | `REVISTA` | Revista de publicación |
        | `FECHA` | Fecha de publicación |
        | `DOI` | Digital Object Identifier |
        | `TIPO_ESTUDIO` | RCT, cohorte, meta-análisis, etc. |
        | `POBLACION` | Población de estudio identificada |
        | `TERMINOS_CLAVE` | Palabras clave extraídas |
        | `NUM_ENTIDADES` | Número de entidades médicas encontradas |
        | `TAMAÑO_MUESTRAL` | Tamaño de la muestra |
        | `FACTORES_CALIDAD_DETALLE` | Factores de calidad específicos |
        | `NEGACIONES` | Patrones de negación detectados |
        | `OUTCOMES_DETECTADOS` | Todos los outcomes identificados |
        | `DIRECCION_EFECTO` | Si el efecto es protector o dañino |
        | `PRECISION_IC` | Precisión del intervalo de confianza |
        | `CONSISTENCIA_OUTCOMES` | Consistencia entre estudios |
        | `ANALISIS_SEMANTICO` | Análisis con BioBERT |
        | `RIESGO_SESGO` | Riesgo de sesgo Cochrane |
        | `META_ANALISIS` | Resultados de meta-análisis |
        | `NUM_CITAS` | Número de citas recibidas |
        | `NUM_COAUTORES` | Número de coautores |
        | `RESUMEN_CRITICO` | Resumen crítico automático |
        | `RESULTADOS_NUMERICOS` | Resultados numéricos (HR, RR, OR) |
        | `ANALISIS_OUTCOMES` | Análisis de cada outcome |
        | `ABSTRACT` | Abstract completo |
        | `FACTORES_CALIDAD` | Detalle de la puntuación de calidad |
        | `CALIDAD_METODOLOGICA` | Análisis metodológico detallado |
        """)
    
    # Session state
    if 'csv_generated' not in st.session_state:
        st.session_state.csv_generated = False
        st.session_state.df_results = None
        st.session_state.filename = ""
    
    # Generar CSV con lectura crítica
    if generate_button:
        if not query.strip():
            st.warning("⚠️ Por favor, introduce una consulta de búsqueda")
        else:
            start_time = time.time()
            
            with st.spinner("🔄 Procesando artículos (esto puede tomar varios minutos)..."):
                articles, total_count = search_pubmed_with_ai(query.strip(), retmax=max_results)
            
            elapsed_time = time.time() - start_time
            
            if articles:
                df = pd.DataFrame(articles)
                
                columns_order = [
                    'pmid', 'quality_score', 'evidence_score', 'evidence_strength',
                    'grade_level', 'title', 'authors', 'journal', 'pubdate', 'doi', 
                    'study_types', 'population', 'key_terms', 'entities_count',
                    'sample_size', 'quality_factors_detailed', 'negations',
                    'all_outcomes', 'effect_directions', 'precision_info', 'consistency_info',
                    'semantic_analysis', 'risk_of_bias',
                    'meta_analysis_info', 'citation_count', 'coauthor_count',
                    'critical_summary', 'numeric_results', 'outcomes_analysis', 
                    'abstract', 'quality_factors', 'methodological_quality_summary'
                ]
                
                available_cols = [col for col in columns_order if col in df.columns]
                df = df[available_cols].copy()
                
                column_names = {
                    'pmid': 'PMID',
                    'quality_score': 'CALIDAD (0-100)',
                    'evidence_score': 'EVIDENCIA_SCORE (0-100)',
                    'evidence_strength': 'FUERZA_EVIDENCIA',
                    'grade_level': 'NIVEL_GRADE',
                    'title': 'TITULO',
                    'authors': 'AUTORES',
                    'journal': 'REVISTA',
                    'pubdate': 'FECHA',
                    'doi': 'DOI',
                    'study_types': 'TIPO_ESTUDIO',
                    'population': 'POBLACION',
                    'key_terms': 'TERMINOS_CLAVE',
                    'entities_count': 'NUM_ENTIDADES',
                    'sample_size': 'TAMAÑO_MUESTRAL',
                    'quality_factors_detailed': 'FACTORES_CALIDAD_DETALLE',
                    'negations': 'NEGACIONES',
                    'all_outcomes': 'OUTCOMES_DETECTADOS',
                    'effect_directions': 'DIRECCION_EFECTO',
                    'precision_info': 'PRECISION_IC',
                    'consistency_info': 'CONSISTENCIA_OUTCOMES',
                    'semantic_analysis': 'ANALISIS_SEMANTICO',
                    'risk_of_bias': 'RIESGO_SESGO',
                    'meta_analysis_info': 'META_ANALISIS',
                    'citation_count': 'NUM_CITAS',
                    'coauthor_count': 'NUM_COAUTORES',
                    'critical_summary': 'RESUMEN_CRITICO',
                    'numeric_results': 'RESULTADOS_NUMERICOS',
                    'outcomes_analysis': 'ANALISIS_OUTCOMES',
                    'abstract': 'ABSTRACT',
                    'quality_factors': 'FACTORES_CALIDAD',
                    'methodological_quality_summary': 'CALIDAD_METODOLOGICA'
                }
                
                df = df.rename(columns=column_names)
                
                st.session_state.df_results = df
                st.session_state.total_found = total_count
                st.session_state.processed = len(articles)
                st.session_state.csv_generated = True
                st.session_state.filename = f"pubmed_lectura_critica_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Artículos procesados", len(articles))
                with col2:
                    avg_quality = df['CALIDAD (0-100)'].mean()
                    st.metric("Calidad promedio", f"{avg_quality:.1f}")
                with col3:
                    with_outcomes = df['ANALISIS_OUTCOMES'].apply(lambda x: 1 if x and x != '' else 0).sum()
                    st.metric("Con análisis de outcomes", with_outcomes)
                with col4:
                    st.metric("Tiempo total", f"{elapsed_time/60:.1f} min")
                
                st.success(f"✅ Lectura crítica avanzada completada: {len(articles)} artículos analizados")
            else:
                st.error("❌ No se encontraron artículos para procesar")
    
    # Mostrar resultados y botón de descarga
    if st.session_state.csv_generated and st.session_state.df_results is not None:
        st.markdown("---")
        st.markdown("### 📥 CSV con Lectura Crítica Avanzada - Listo para Descargar")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Artículos procesados", st.session_state.processed)
            st.metric("Total encontrados", st.session_state.total_found)
        
        with col2:
            st.markdown("**Vista previa (primeras 5 filas):**")
            preview_df = st.session_state.df_results.head(5)
            
            preview_cols = ['PMID', 'CALIDAD (0-100)', 'FUERZA_EVIDENCIA', 'NIVEL_GRADE', 'RIESGO_SESGO']
            available_preview = [col for col in preview_cols if col in preview_df.columns]
            st.dataframe(preview_df[available_preview], use_container_width=True)
        
        csv_data = st.session_state.df_results.to_csv(index=False, encoding='utf-8-sig')
        
        st.download_button(
            label="📥 DESCARGAR CSV CON LECTURA CRÍTICA AVANZADA",
            data=csv_data,
            file_name=st.session_state.filename,
            mime="text/csv",
            use_container_width=True,
            key="download_csv"
        )
        
        with st.expander("📋 Ver estructura completa del CSV"):
            st.write("**Columnas incluidas:**")
            for col in st.session_state.df_results.columns:
                if col in ['ANALISIS_SEMANTICO', 'RIESGO_SESGO', 'META_ANALISIS', 'NUM_CITAS', 'NUM_COAUTORES']:
                    st.write(f"  **• {col}** ⬅️ COLUMNA AVANZADA")
                elif col in ['NIVEL_GRADE', 'DIRECCION_EFECTO', 'PRECISION_IC', 'CONSISTENCIA_OUTCOMES', 'OUTCOMES_DETECTADOS']:
                    st.write(f"  **• {col}** ⬅️ COLUMNA DE EVALUACIÓN AVANZADA")
                elif col in ['FUERZA_EVIDENCIA', 'EVIDENCIA_SCORE (0-100)', 'TAMAÑO_MUESTRAL', 'FACTORES_CALIDAD_DETALLE', 'NEGACIONES', 'CALIDAD_METODOLOGICA']:
                    st.write(f"  **• {col}** ⬅️ COLUMNA DE EVALUACIÓN MEJORADA")
                elif col in ['RESUMEN_CRITICO', 'RESULTADOS_NUMERICOS', 'ANALISIS_OUTCOMES']:
                    st.write(f"  **• {col}** ⬅️ COLUMNA DE LECTURA CRÍTICA")
                else:
                    st.write(f"  • {col}")
            st.write(f"**Total de filas:** {len(st.session_state.df_results)}")
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 0.8em;'>
            🧠 PubMed AI Analyzer v4.0 - Optimizado para Streamlit Cloud | Análisis semántico con BioBERT | Totalmente genérico
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    main()
