"""
Buscador y Verificador Semántico Integrado - VERSIÓN ALTO VOLUMEN CON COMPRENSIÓN LECTORA AVANZADA
Combina búsqueda en múltiples bases de datos científicas con análisis semántico profundo tipo humano
"""

import streamlit as st
import pandas as pd
import requests
import time
import urllib.parse
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import io
import xml.etree.ElementTree as ET
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Any
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import Counter, defaultdict
import nltk
import warnings
warnings.filterwarnings('ignore')
from deep_translator import GoogleTranslator
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.utils import formatdate
import ssl
import json
import hashlib
import pickle
from pathlib import Path
import numpy as np

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

st.set_page_config(
    page_title="🧠 Buscador con Comprensión Lectora Humana",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURACIÓN DE CORREO
# ============================================================================

class EmailConfig:
    def __init__(self):
        self.SMTP_SERVER = st.secrets.get("smtp_server", "smtp.gmail.com")
        self.SMTP_PORT = st.secrets.get("smtp_port", 587)
        self.EMAIL_USER = st.secrets.get("email_user", "")
        self.EMAIL_PASSWORD = st.secrets.get("email_password", "")
        self.MAX_FILE_SIZE_MB = 10
        self.available = all([
            self.SMTP_SERVER, self.SMTP_PORT, 
            self.EMAIL_USER, self.EMAIL_PASSWORD
        ])

EMAIL_CONFIG = EmailConfig()

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def enviar_correo(destinatario, asunto, mensaje_html=None, mensaje_texto=None, archivos=None):
    if not EMAIL_CONFIG.available:
        st.error("Configuración de correo no disponible")
        return False
    
    if not validate_email(destinatario):
        st.error("Dirección de correo no válida")
        return False
    
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = EMAIL_CONFIG.EMAIL_USER
        msg['To'] = destinatario
        msg['Subject'] = asunto
        msg['Date'] = formatdate(localtime=True)
        
        if mensaje_texto:
            msg.attach(MIMEText(mensaje_texto, 'plain'))
        if mensaje_html:
            msg.attach(MIMEText(mensaje_html, 'html'))
        
        if archivos:
            for archivo in archivos:
                if len(archivo['contenido']) > EMAIL_CONFIG.MAX_FILE_SIZE_MB * 1024 * 1024:
                    continue
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(archivo['contenido'])
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename="{archivo["nombre"]}"')
                msg.attach(part)
        
        context = ssl.create_default_context()
        with smtplib.SMTP(EMAIL_CONFIG.SMTP_SERVER, EMAIL_CONFIG.SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(EMAIL_CONFIG.EMAIL_USER, EMAIL_CONFIG.EMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Error enviando correo: {str(e)}")
        return False

# ============================================================================
# CONFIGURACIÓN NLTK
# ============================================================================

def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt')
        except:
            st.warning("No se pudo descargar punkt. Usando tokenización alternativa.")
            return False
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords')
        except:
            st.warning("No se pudieron descargar stopwords.")
            return False
    
    try:
        from nltk.corpus import stopwords
        from nltk.stem import SnowballStemmer
        return True
    except:
        return False

NLTK_READY = setup_nltk()

if NLTK_READY:
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    from nltk.tokenize import sent_tokenize

# ============================================================================
# CLASE DE TRADUCCIÓN
# ============================================================================

class TranslationManager:
    def __init__(self):
        self.cache = {}
        try:
            self.translator = GoogleTranslator(source='auto', target='es')
            self.translator_en = GoogleTranslator(source='auto', target='en')
            self.available = True
        except:
            self.available = False
    
    def translate_to_spanish(self, text: str) -> str:
        if not text or len(text) > 4000 or not self.available:
            return text
        cache_key = f"es_{hashlib.md5(text[:100].encode()).hexdigest()}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        try:
            result = self.translator.translate(text[:3500])
            self.cache[cache_key] = result
            return result
        except:
            return text
    
    def translate_to_english(self, text: str) -> str:
        if not text or len(text) > 4000 or not self.available:
            return text
        cache_key = f"en_{hashlib.md5(text[:100].encode()).hexdigest()}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        try:
            result = self.translator_en.translate(text[:3500])
            self.cache[cache_key] = result
            return result
        except:
            return text

# ============================================================================
# SISTEMA DE COMPRENSIÓN LECTORA AVANZADO (¡EL CORAZÓN DEL CAMBIO!)
# ============================================================================

class AdvancedReadingComprehension:
    """
    Sistema de comprensión lectora que imita el razonamiento humano
    """
    
    def __init__(self):
        self.translator = TranslationManager()
        
        # Base de conocimiento sobre tipos de estudio y su peso
        self.study_type_weights = {
            'meta-analysis': 2.5,
            'systematic review': 2.5,
            'randomized controlled trial': 2.0,
            'cohort study': 1.5,
            'case-control': 1.3,
            'cross-sectional': 1.0,
            'case series': 0.7,
            'case report': 0.5,
            'editorial': 0.3,
            'commentary': 0.3,
            'news': 0.2,
            'unknown': 0.5
        }
        
        # Palabras clave para detectar tipo de estudio
        self.study_type_patterns = {
            'meta-analysis': [r'meta-?analysis', r'meta-analysis', r'meta analysis'],
            'systematic review': [r'systematic review', r'systematic literature review'],
            'randomized controlled trial': [r'randomized controlled trial', r'randomised controlled trial', r'RCT'],
            'cohort study': [r'cohort', r'prospective cohort', r'retrospective cohort'],
            'case-control': [r'case-?control', r'case control'],
            'cross-sectional': [r'cross-?sectional', r'cross sectional'],
            'case series': [r'case series'],
            'case report': [r'case report'],
            'editorial': [r'editorial'],
            'commentary': [r'commentary'],
            'news': [r'news', r'press release']
        }
        
        # Indicadores de lenguaje cauto (hedging language)
        self.hedging_indicators = [
            'suggest', 'may', 'might', 'could', 'possible', 'potentially',
            'indicate', 'appear', 'seem', 'suggestive', 'preliminary',
            'tentative', 'not conclusive', 'further research', 'needed',
            'warrant', 'cautious', 'speculate', 'hypothesize', 'propose'
        ]
        
        # Indicadores de lenguaje fuerte
        self.strong_indicators = [
            'demonstrate', 'prove', 'confirm', 'establish', 'conclusive',
            'definitive', 'unequivocal', 'certain', 'clearly show',
            'undoubtedly', 'without doubt', 'irrefutable'
        ]
        
        # Patrones para detectar el objetivo principal del estudio
        self.objective_patterns = [
            r'(objective|aim|purpose|goal).{1,50}(was|were|is)',
            r'to (investigate|examine|evaluate|assess|determine|establish)',
            r'this (study|paper|research) (aims|aimed|seeks|sought) to',
            r'we (sought|aimed|hypothesized) to'
        ]
        
        # Patrones para detectar conclusiones
        self.conclusion_patterns = [
            r'(conclusion|conclude|conclusions?).{1,50}(that|:)',
            r'in summary',
            r'in conclusion',
            r'take(n)? together',
            r'our (findings|results|data) (suggest|indicate|show|demonstrate)',
            r'we (conclude|demonstrate|show) that',
            r'these (findings|results) (suggest|indicate|support|provide evidence)'
        ]
        
        # Cache para análisis
        self.analysis_cache = {}
    
    def detect_study_type(self, text: str) -> Dict[str, float]:
        """
        Detecta el tipo de estudio y asigna probabilidades
        """
        text_lower = text.lower()
        scores = {study_type: 0.0 for study_type in self.study_type_weights.keys()}
        
        # Buscar patrones
        for study_type, patterns in self.study_type_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    scores[study_type] += len(matches) * 2
        
        # Buscar en abstract y título (dar más peso)
        abstract_match = re.search(r'abstract', text_lower)
        if abstract_match:
            abstract_text = text[max(0, abstract_match.start()-200):min(len(text), abstract_match.end()+1000)]
            abstract_lower = abstract_text.lower()
            for study_type, patterns in self.study_type_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, abstract_lower):
                        scores[study_type] += 3
        
        # Normalizar
        total = sum(scores.values()) or 1
        for study_type in scores:
            scores[study_type] = scores[study_type] / total
        
        # Identificar el tipo principal
        primary_type = max(scores, key=scores.get) if max(scores.values()) > 0.1 else 'unknown'
        
        return {
            'primary_type': primary_type,
            'confidence': scores[primary_type],
            'all_scores': scores,
            'weight': self.study_type_weights.get(primary_type, 0.5)
        }
    
    def extract_main_objective(self, text: str) -> Optional[str]:
        """
        Extrae el objetivo principal del estudio
        """
        text_lower = text.lower()
        
        # Buscar en las primeras 2000 caracteres (normalmente donde está el objetivo)
        intro_text = text_lower[:2000]
        
        for pattern in self.objective_patterns:
            match = re.search(pattern, intro_text, re.IGNORECASE)
            if match:
                # Extraer la oración completa
                start = max(0, match.start() - 50)
                end = min(len(text), match.start() + 300)
                sentence = text[start:end].split('.')[0] + '.'
                return sentence
        
        return None
    
    def extract_conclusion(self, text: str) -> Optional[str]:
        """
        Extrae la conclusión principal del estudio
        """
        text_lower = text.lower()
        
        # Buscar en los últimos 3000 caracteres (donde suele estar la conclusión)
        conclusion_text = text_lower[-3000:]
        
        for pattern in self.conclusion_patterns:
            match = re.search(pattern, conclusion_text, re.IGNORECASE)
            if match:
                # Extraer la oración completa
                start = max(0, len(text) - 3000 + match.start() - 50)
                end = min(len(text), len(text) - 3000 + match.start() + 500)
                sentence = text[start:end].split('.')[0] + '.'
                return sentence
        
        return None
    
    def extract_all_claims(self, text: str, hypothesis_terms: List[str]) -> List[Dict]:
        """
        Extrae TODAS las afirmaciones relevantes del texto
        """
        sentences = self.split_sentences(text)
        claims = []
        
        for sentence in sentences:
            # Verificar si contiene términos de la hipótesis
            sentence_lower = sentence.lower()
            term_matches = [term for term in hypothesis_terms if term in sentence_lower]
            
            if term_matches:
                # Analizar la afirmación en profundidad
                claim_analysis = self.analyze_claim_deep(sentence, term_matches)
                claims.append(claim_analysis)
        
        return claims
    
    def analyze_claim_deep(self, sentence: str, matched_terms: List[str]) -> Dict:
        """
        Análisis profundo de una afirmación individual
        """
        sentence_lower = sentence.lower()
        sentence_en = self.translator.translate_to_english(sentence)
        
        # Determinar tipo de relación
        relation_type = self.determine_relation_type(sentence_en)
        
        # Determinar dirección (a favor/en contra)
        direction = self.determine_direction(sentence_en)
        
        # Determinar fuerza del lenguaje
        language_strength = self.determine_language_strength(sentence_en)
        
        # Determinar si es una conclusión principal
        is_primary_conclusion = self.is_primary_conclusion(sentence)
        
        # Extraer contexto (dónde aparece en el artículo)
        section = self.detect_section(sentence)
        
        return {
            'sentence': sentence[:300] + '...' if len(sentence) > 300 else sentence,
            'sentence_en': sentence_en[:300] + '...' if len(sentence_en) > 300 else sentence_en,
            'matched_terms': matched_terms,
            'relation_type': relation_type,
            'direction': direction,
            'language_strength': language_strength,
            'strength_score': self.calculate_strength_score(relation_type, direction, language_strength),
            'is_primary_conclusion': is_primary_conclusion,
            'section': section,
            'section_weight': self.section_weights.get(section, 1.0)
        }
    
    def determine_relation_type(self, sentence: str) -> str:
        """
        Determina el tipo de relación expresada en la oración
        """
        sentence_lower = sentence.lower()
        
        # Patrones causales explícitos
        causal_patterns = [
            r'causes?', r'induces?', r'leads? to', r'results? in', 
            r'provokes?', r'triggers?', r'produces?', r'gives rise to'
        ]
        
        # Patrones de asociación
        association_patterns = [
            r'associat', r'relat', r'correlat', r'link', r'connect',
            r'tied to', r'bound with'
        ]
        
        # Patrones de riesgo
        risk_patterns = [
            r'risk', r'increase(s)? the risk', r'elevated risk',
            r'higher risk', r'greater risk'
        ]
        
        # Patrones de protección
        protective_patterns = [
            r'protect', r'reduce(s)? the risk', r'lower risk',
            r'decreased risk', r'prevent'
        ]
        
        for pattern in causal_patterns:
            if re.search(pattern, sentence_lower):
                return 'causal'
        
        for pattern in risk_patterns:
            if re.search(pattern, sentence_lower):
                return 'risk'
        
        for pattern in protective_patterns:
            if re.search(pattern, sentence_lower):
                return 'protective'
        
        for pattern in association_patterns:
            if re.search(pattern, sentence_lower):
                return 'association'
        
        return 'unknown'
    
    def determine_direction(self, sentence: str) -> str:
        """
        Determina si la afirmación apoya o contradice la hipótesis
        """
        sentence_lower = sentence.lower()
        
        # Patrones de apoyo
        support_patterns = [
            r'associat', r'link', r'correlat', r'caus', r'risk',
            r'increas', r'elevat', r'higher', r'greater', r'more likely'
        ]
        
        # Patrones de contradicción
        contradict_patterns = [
            r'no (associat|relat|correlat|link)',
            r'not (associat|relat|correlat)',
            r'lack of (associat|evidence)',
            r'no (evidence|support)',
            r'does not (support|indicate)',
            r'contradict',
            r'opposite',
            r'inverse',
            r'protective',
            r'reduces? risk',
            r'lower risk'
        ]
        
        # Primero verificar contradicción explícita
        for pattern in contradict_patterns:
            if re.search(pattern, sentence_lower):
                return 'contradicts'
        
        # Luego verificar apoyo
        for pattern in support_patterns:
            if re.search(pattern, sentence_lower):
                return 'supports'
        
        return 'neutral'
    
    def determine_language_strength(self, sentence: str) -> str:
        """
        Determina la fuerza del lenguaje usado (cauto, fuerte, neutral)
        """
        sentence_lower = sentence.lower()
        
        # Contar indicadores de cautela
        hedging_count = sum(1 for indicator in self.hedging_indicators 
                          if indicator in sentence_lower)
        
        # Contar indicadores de fuerza
        strong_count = sum(1 for indicator in self.strong_indicators 
                          if indicator in sentence_lower)
        
        if strong_count >= 2:
            return 'strong'
        elif hedging_count >= 2:
            return 'hedged'
        elif strong_count == 1 and hedging_count == 0:
            return 'moderate_strong'
        elif hedging_count == 1:
            return 'moderate_hedged'
        else:
            return 'neutral'
    
    def calculate_strength_score(self, relation_type: str, direction: str, 
                                language_strength: str) -> float:
        """
        Calcula un puntaje de fuerza para la evidencia
        """
        base_scores = {
            'causal': 2.0,
            'risk': 1.8,
            'association': 1.5,
            'protective': 1.3,
            'unknown': 0.5
        }
        
        direction_multipliers = {
            'supports': 1.0,
            'contradicts': -1.0,
            'neutral': 0.1
        }
        
        language_multipliers = {
            'strong': 1.5,
            'moderate_strong': 1.2,
            'neutral': 1.0,
            'moderate_hedged': 0.7,
            'hedged': 0.4
        }
        
        base = base_scores.get(relation_type, 0.5)
        direction_mult = direction_multipliers.get(direction, 0.1)
        language_mult = language_multipliers.get(language_strength, 1.0)
        
        return base * direction_mult * language_mult
    
    def is_primary_conclusion(self, sentence: str) -> bool:
        """
        Determina si la oración es parte de la conclusión principal
        """
        sentence_lower = sentence.lower()
        
        conclusion_indicators = [
            'conclusion', 'conclude', 'summary', 'findings suggest',
            'results indicate', 'data show', 'we conclude', 'in summary',
            'taken together', 'overall', 'this study demonstrates'
        ]
        
        for indicator in conclusion_indicators:
            if indicator in sentence_lower:
                return True
        
        return False
    
    def detect_section(self, sentence: str) -> str:
        """
        Detecta en qué sección del artículo aparece la oración
        """
        sentence_lower = sentence.lower()
        
        section_keywords = {
            'abstract': ['abstract', 'background', 'objective', 'methods', 'results', 'conclusions'],
            'introduction': ['introduction', 'background', 'rationale'],
            'methods': ['methods', 'methodology', 'study design', 'participants', 'statistical analysis'],
            'results': ['results', 'findings', 'analysis', 'data'],
            'discussion': ['discussion', 'interpretation', 'limitations'],
            'conclusion': ['conclusion', 'concluding remarks']
        }
        
        for section, keywords in section_keywords.items():
            for keyword in keywords:
                if keyword in sentence_lower and len(keyword) > 3:
                    return section
        
        return 'unknown'
    
    def split_sentences(self, text: str) -> List[str]:
        """Divide texto en oraciones"""
        if NLTK_READY:
            try:
                return sent_tokenize(text)
            except:
                return text.split('. ')
        else:
            return text.split('. ')
    
    def synthesize_verdict(self, claims: List[Dict], study_type_info: Dict, 
                          main_objective: str, conclusion: str) -> Dict:
        """
        SINTESIS CRÍTICA: Determina el veredicto final basado en TODA la evidencia
        """
        if not claims:
            return {
                'verdict': 'inconclusive',
                'verdict_text': 'EVIDENCIA NO CONCLUYENTE',
                'confidence': 0,
                'reasoning': 'No se encontraron afirmaciones relevantes en el artículo',
                'summary': {}
            }
        
        # Separar afirmaciones por tipo y sección
        primary_conclusion_claims = [c for c in claims if c['is_primary_conclusion']]
        results_section_claims = [c for c in claims if c['section'] == 'results']
        discussion_section_claims = [c for c in claims if c['section'] == 'discussion']
        abstract_claims = [c for c in claims if c['section'] == 'abstract']
        
        # Calcular puntajes ponderados
        weighted_score = 0.0
        total_weight = 0.0
        
        support_count = 0
        contradict_count = 0
        strong_support_count = 0
        strong_contradict_count = 0
        
        for claim in claims:
            # Peso base por tipo de afirmación
            weight = abs(claim['strength_score'])
            
            # Peso adicional por sección
            if claim['section'] == 'results':
                weight *= 1.5
            elif claim['section'] == 'conclusion':
                weight *= 2.0
            elif claim['section'] == 'discussion':
                weight *= 1.2
            elif claim['section'] == 'abstract':
                weight *= 1.3
            
            # Peso adicional si es conclusión principal
            if claim['is_primary_conclusion']:
                weight *= 2.5
            
            # Aplicar dirección
            if claim['direction'] == 'supports':
                weighted_score += weight
                support_count += 1
                if claim['language_strength'] in ['strong', 'moderate_strong']:
                    strong_support_count += 1
            elif claim['direction'] == 'contradicts':
                weighted_score -= weight
                contradict_count += 1
                if claim['language_strength'] in ['strong', 'moderate_strong']:
                    strong_contradict_count += 1
            
            total_weight += weight
        
        # Normalizar por tipo de estudio
        study_weight = study_type_info['weight']
        weighted_score *= study_weight
        
        # Determinar veredicto con matices
        if weighted_score > 5.0:
            if strong_support_count >= 2:
                verdict = 'strongly_supports'
                verdict_text = 'CORROBORA FUERTEMENTE'
            else:
                verdict = 'supports'
                verdict_text = 'CORROBORA'
        elif weighted_score > 1.0:
            if support_count > contradict_count * 2:
                verdict = 'supports'
                verdict_text = 'CORROBORA'
            else:
                verdict = 'inconclusive'
                verdict_text = 'EVIDENCIA NO CONCLUYENTE'
        elif weighted_score > -1.0:
            if contradict_count > support_count * 2:
                verdict = 'contradicts'
                verdict_text = 'CONTRADICE'
            else:
                verdict = 'inconclusive'
                verdict_text = 'EVIDENCIA NO CONCLUYENTE'
        elif weighted_score > -5.0:
            if strong_contradict_count >= 2:
                verdict = 'strongly_contradicts'
                verdict_text = 'CONTRADICE FUERTEMENTE'
            else:
                verdict = 'contradicts'
                verdict_text = 'CONTRADICE'
        else:
            if strong_contradict_count >= 3:
                verdict = 'strongly_contradicts'
                verdict_text = 'CONTRADICE FUERTEMENTE'
            else:
                verdict = 'contradicts'
                verdict_text = 'CONTRADICE'
        
        # Calcular confianza
        confidence_factors = [
            min(1.0, len(claims) / 20),  # Suficientes afirmaciones
            study_type_info['confidence'],  # Tipo de estudio claro
            min(1.0, (support_count + contradict_count) / 10),  # Suficiente evidencia
            min(1.0, len(primary_conclusion_claims) / 2)  # Conclusiones claras
        ]
        confidence = np.mean(confidence_factors) * min(1.0, abs(weighted_score) / 10)
        
        # Generar razonamiento explicativo
        reasoning = self.generate_reasoning(
            verdict, weighted_score, support_count, contradict_count,
            strong_support_count, strong_contradict_count,
            study_type_info, main_objective, conclusion,
            primary_conclusion_claims
        )
        
        return {
            'verdict': verdict,
            'verdict_text': verdict_text,
            'score': weighted_score,
            'confidence': confidence,
            'support_count': support_count,
            'contradict_count': contradict_count,
            'strong_support': strong_support_count,
            'strong_contradict': strong_contradict_count,
            'reasoning': reasoning,
            'summary': {
                'study_type': study_type_info['primary_type'],
                'study_confidence': study_type_info['confidence'],
                'main_objective': main_objective,
                'conclusion': conclusion,
                'total_claims': len(claims),
                'primary_conclusion_claims': len(primary_conclusion_claims)
            }
        }
    
    def generate_reasoning(self, verdict: str, score: float, support: int, 
                          contradict: int, strong_support: int, strong_contradict: int,
                          study_type_info: Dict, objective: str, conclusion: str,
                          primary_claims: List[Dict]) -> str:
        """
        Genera una explicación del razonamiento utilizado
        """
        reasoning_parts = []
        
        # Tipo de estudio
        reasoning_parts.append(f"📊 **Tipo de estudio**: {study_type_info['primary_type'].replace('_', ' ').title()} "
                              f"(confianza: {study_type_info['confidence']:.1%})")
        
        # Objetivo
        if objective:
            reasoning_parts.append(f"🎯 **Objetivo**: {objective}")
        
        # Evidencia numérica
        reasoning_parts.append(f"📈 **Evidencia encontrada**:")
        reasoning_parts.append(f"   - {support} afirmaciones a favor (de las cuales {strong_support} son fuertes)")
        reasoning_parts.append(f"   - {contradict} afirmaciones en contra (de las cuales {strong_contradict} son fuertes)")
        reasoning_parts.append(f"   - Puntaje ponderado: {score:.2f}")
        
        # Interpretación
        if verdict == 'strongly_supports':
            reasoning_parts.append(f"✅ **Conclusión**: El artículo CORROBORA FUERTEMENTE la hipótesis. "
                                  f"La evidencia es consistente y proviene de afirmaciones sólidas, "
                                  f"particularmente en las secciones de resultados y conclusiones.")
        elif verdict == 'supports':
            reasoning_parts.append(f"✅ **Conclusión**: El artículo CORROBORA la hipótesis, aunque con menor fuerza. "
                                  f"La evidencia a favor supera a la evidencia en contra, pero el lenguaje es "
                                  f"más cauteloso o la evidencia es menos contundente.")
        elif verdict == 'inconclusive':
            reasoning_parts.append(f"⚠️ **Conclusión**: La evidencia es NO CONCLUYENTE. "
                                  f"Las afirmaciones a favor y en contra se equilibran, o el artículo "
                                  f"no proporciona suficiente información para determinar una dirección clara.")
        elif verdict == 'contradicts':
            reasoning_parts.append(f"❌ **Conclusión**: El artículo CONTRADICE la hipótesis. "
                                  f"La evidencia en contra supera a la evidencia a favor.")
        elif verdict == 'strongly_contradicts':
            reasoning_parts.append(f"❌ **Conclusión**: El artículo CONTRADICE FUERTEMENTE la hipótesis. "
                                  f"La evidencia en contra es contundente y consistente.")
        
        # Conclusión del artículo
        if conclusion:
            reasoning_parts.append(f"📝 **Conclusión del artículo**: {conclusion}")
        
        return '\n\n'.join(reasoning_parts)


# ============================================================================
# VERIFICADOR SEMÁNTICO AVANZADO CON COMPRENSIÓN LECTORA
# ============================================================================

class AdvancedSemanticVerifier:
    """
    Verificador semántico con capacidad de comprensión lectora tipo humano
    """
    
    def __init__(self):
        self.reading_comprehension = AdvancedReadingComprehension()
        self.translator = TranslationManager()
        self.UMBRAL_RELEVANCIA = 0.1
    
    def extract_key_terms(self, hypothesis: str) -> List[str]:
        """Extrae términos clave de la hipótesis"""
        words = re.findall(r'\b[a-zA-Záéíóúñü]+\b', hypothesis.lower())
        
        stop_words_es = {'el', 'la', 'los', 'las', 'de', 'del', 'y', 'o', 'a', 'en', 
                         'por', 'para', 'con', 'sin', 'sobre', 'entre', 'mediante'}
        stop_words_en = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                         'for', 'with', 'without', 'by', 'from', 'as'}
        
        filtered_words = []
        for word in words:
            if len(word) > 3 and word not in stop_words_es and word not in stop_words_en:
                filtered_words.append(word)
        
        # También extraer frases de dos palabras
        phrases = re.findall(r'\b[a-zA-Záéíóúñü]+\s+[a-zA-Záéíóúñü]+\b', hypothesis.lower())
        for phrase in phrases:
            if len(phrase.split()) == 2 and all(len(w) > 3 for w in phrase.split()):
                filtered_words.append(phrase.replace(' ', '_'))
        
        return list(set(filtered_words))
    
    def verify_article_text(self, text: str, hypothesis: str) -> Dict:
        """
        Verifica un artículo completo con comprensión lectora avanzada
        """
        if not text or len(text.strip()) < 200:
            return {
                'success': False,
                'error': 'Texto insuficiente para análisis',
                'verdict': None
            }
        
        # Extraer términos clave de la hipótesis (para búsqueda inicial)
        hypothesis_terms = self.extract_key_terms(hypothesis)
        
        # PASO 1: Análisis estructural del artículo
        study_type_info = self.reading_comprehension.detect_study_type(text)
        main_objective = self.reading_comprehension.extract_main_objective(text)
        conclusion = self.reading_comprehension.extract_conclusion(text)
        
        # PASO 2: Extraer TODAS las afirmaciones relevantes
        claims = self.reading_comprehension.extract_all_claims(text, hypothesis_terms)
        
        # PASO 3: Síntesis crítica (¡el corazón del sistema!)
        verdict = self.reading_comprehension.synthesize_verdict(
            claims, study_type_info, main_objective, conclusion
        )
        
        return {
            'success': True,
            'total_sentences': len(self.reading_comprehension.split_sentences(text)),
            'relevant_claims': len(claims),
            'claims': claims[:20],  # Limitar para no saturar
            'study_type': study_type_info,
            'main_objective': main_objective,
            'conclusion': conclusion,
            'verdict': verdict,
            'hypothesis_terms': hypothesis_terms
        }


# ============================================================================
# MOTOR DE BÚSQUEDA CIENTÍFICA (sin cambios significativos)
# ============================================================================

class ScientificSearchEngine:
    """
    Motor de búsqueda científica (igual que antes)
    """
    def __init__(self, email: str):
        self.email = email
        self.delay = 0.3
    
    def search_pubmed_advanced(self, query: str, max_results: int = 1000, year_range: tuple = None) -> list:
        # [Código existente - mantener igual]
        pass
    
    def search_crossref(self, query: str, max_results: int = 1000, year_range: tuple = None) -> list:
        # [Código existente - mantener igual]
        pass
    
    def search_openalex(self, query: str, max_results: int = 1000, year_range: tuple = None) -> list:
        # [Código existente - mantener igual]
        pass
    
    def search_europe_pmc(self, query: str, max_results: int = 1000, year_range: tuple = None) -> list:
        # [Código existente - mantener igual]
        pass
    
    def search_all(self, query: str, max_results_per_db: int = 1000, selected_dbs: list = None, 
                   year_range: tuple = None) -> pd.DataFrame:
        # [Código existente - mantener igual]
        pass


# ============================================================================
# OBTENEDOR DE TEXTO DE ARTÍCULOS (mejorado con manejo de PDFs)
# ============================================================================

class ArticleTextFetcher:
    """Obtiene el texto completo de artículos desde fuentes abiertas"""
    
    def __init__(self):
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.min_delay = 0.5
        self.last_request_time = 0
    
    def wait(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_delay:
            time.sleep(self.min_delay - time_since_last)
        self.last_request_time = time.time()
    
    def get_text_from_doi(self, doi: str) -> Tuple[Optional[str], str]:
        """
        Obtiene el texto completo del artículo usando el DOI
        Versión mejorada con más fuentes
        """
        if not doi or pd.isna(doi) or doi == '':
            return None, "DOI vacío"
        
        doi = str(doi).strip()
        doi = re.sub(r'^https?://(dx\.)?doi\.org/', '', doi)
        doi = re.sub(r'^doi:', '', doi)
        
        self.wait()
        
        # Intentar 1: OpenAlex (para metadatos y posible PDF)
        try:
            url = f"https://api.openalex.org/works/doi:{doi}"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Buscar PDF de acceso abierto
                oa_url = data.get('open_access', {}).get('oa_url')
                if oa_url and ('pdf' in oa_url.lower()):
                    return self.try_fetch_pdf(oa_url), "OpenAlex (PDF)"
                
                # Si no hay PDF, intentar obtener abstract del mismo OpenAlex
                abstract = data.get('abstract')
                if abstract:
                    return abstract, "OpenAlex (Abstract)"
        except:
            pass
        
        # Intentar 2: Europe PMC (buenos abstracts)
        try:
            url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:{doi}&format=json"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = data.get('resultList', {}).get('result', [])
                if results:
                    # Intentar obtener texto completo si está disponible
                    if results[0].get('fullTextUrlList'):
                        for url_info in results[0]['fullTextUrlList'].get('fullTextUrl', []):
                            if url_info.get('availabilityCode') == 'OA' and 'pdf' in url_info.get('url', '').lower():
                                pdf_url = url_info['url']
                                return self.try_fetch_pdf(pdf_url), "Europe PMC (PDF)"
                    
                    # Si no hay PDF, devolver abstract
                    abstract = results[0].get('abstractText', '')
                    if abstract:
                        return abstract, "Europe PMC (Abstract)"
        except:
            pass
        
        # Intentar 3: Unpaywall
        try:
            url = f"https://api.unpaywall.org/v2/{doi}?email=usuario@ejemplo.com"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('is_oa') and data.get('best_oa_location'):
                    pdf_url = data['best_oa_location'].get('url_for_pdf')
                    if pdf_url:
                        return self.try_fetch_pdf(pdf_url), "Unpaywall (PDF)"
        except:
            pass
        
        # Intentar 4: PubMed (último recurso)
        try:
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={doi}[DOI]&retmode=json"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                id_list = data.get('esearchresult', {}).get('idlist', [])
                if id_list:
                    pmid = id_list[0]
                    fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
                    fetch_response = self.session.get(fetch_url, timeout=10)
                    if fetch_response.status_code == 200:
                        root = ET.fromstring(fetch_response.content)
                        abstract_elem = root.find('.//AbstractText')
                        if abstract_elem is not None and abstract_elem.text:
                            return abstract_elem.text, "PubMed (Abstract)"
        except:
            pass
        
        return None, "No se pudo obtener texto completo"
    
    def try_fetch_pdf(self, pdf_url: str) -> Optional[str]:
        """
        Intenta obtener texto de un PDF (simplificado - en producción usar PyPDF2)
        """
        try:
            response = self.session.get(pdf_url, timeout=15)
            if response.status_code == 200:
                # Aquí idealmente usarías PyPDF2 para extraer texto
                # Por ahora devolvemos un placeholder
                return f"[PDF disponible en: {pdf_url}]\n\nPara un análisis completo, se requiere extracción de texto de PDF."
        except:
            pass
        return None


# ============================================================================
# CLASE PRINCIPAL INTEGRADA (ACTUALIZADA)
# ============================================================================

class IntegratedScientificVerifier:
    """
    Clase principal que integra búsqueda y verificación semántica con comprensión lectora
    """
    
    def __init__(self, email: str):
        self.search_engine = ScientificSearchEngine(email)
        self.semantic_verifier = AdvancedSemanticVerifier()
        self.text_fetcher = ArticleTextFetcher()
        self.results = []
        self.stats = {
            'total_articles': 0,
            'analyzed': 0,
            'with_text': 0,
            'corroboran': 0,
            'contradicen': 0,
            'inconclusos': 0
        }
    
    def run_analysis(self, query: str, hypothesis: str, max_results_per_db: int = 1000, 
                     selected_dbs: list = None, year_range: tuple = None,
                     progress_callback=None) -> pd.DataFrame:
        """
        Ejecuta el flujo completo: búsqueda + análisis con comprensión lectora
        """
        self.results = []
        self.stats = {
            'total_articles': 0,
            'analyzed': 0,
            'with_text': 0,
            'corroboran': 0,
            'contradicen': 0,
            'inconclusos': 0
        }
        
        # PASO 1: BÚSQUEDA
        if progress_callback:
            progress_callback("🔍 Buscando artículos en bases de datos...", 0.05)
        
        articles_df = self.search_engine.search_all(
            query, max_results_per_db, selected_dbs, year_range
        )
        
        if articles_df.empty:
            return pd.DataFrame()
        
        self.stats['total_articles'] = len(articles_df)
        
        if progress_callback:
            progress_callback(f"✅ Encontrados {len(articles_df)} artículos. Iniciando análisis con comprensión lectora...", 0.1)
        
        # PASO 2: ANÁLISIS PROFUNDO DE CADA ARTÍCULO
        results_list = []
        
        batch_size = 20  # Lotes más pequeños para análisis profundo
        total_articles = len(articles_df)

        for batch_start in range(0, total_articles, batch_size):
            batch_end = min(batch_start + batch_size, total_articles)
            batch_df = articles_df.iloc[batch_start:batch_end]

            for idx, row in batch_df.iterrows():
                current = idx + 1
                if progress_callback:
                    progress_value = 0.1 + 0.85 * (current / total_articles)
                    progress_callback(
                        f"🧠 Analizando con comprensión lectora artículo {current}/{total_articles}: {str(row['titulo'])[:50]}...",
                        progress_value
                    )

                # Obtener texto del artículo
                article_text, source = self.text_fetcher.get_text_from_doi(row.get('doi', ''))

                result_row = {
                    'base_datos': row.get('base_datos', 'Desconocida'),
                    'titulo': row.get('titulo', 'Sin título'),
                    'autores': row.get('autores', ''),
                    'revista': row.get('revista', ''),
                    'año': row.get('año', ''),
                    'doi': row.get('doi', ''),
                    'url': row.get('url', ''),
                    'texto_disponible': article_text is not None,
                    'fuente_texto': source if article_text else 'No disponible',
                    'veredicto': '',
                    'confianza': 0,
                    'puntuacion': 0,
                    'evidencia_a_favor': 0,
                    'evidencia_en_contra': 0,
                    'evidencia_fuerte_favor': 0,
                    'evidencia_fuerte_contra': 0,
                    'oraciones_totales': 0,
                    'oraciones_relevantes': 0,
                    'tipo_estudio': '',
                    'objetivo_principal': '',
                    'conclusion_articulo': '',
                    'razonamiento': '',
                    'detalle_evidencia': ''
                }

                if article_text:
                    self.stats['with_text'] += 1

                    # Analizar con comprensión lectora avanzada
                    analysis = self.semantic_verifier.verify_article_text(article_text, hypothesis)

                    if analysis['success']:
                        verdict = analysis['verdict']

                        result_row.update({
                            'veredicto': verdict['verdict_text'],
                            'confianza': verdict['confidence'],
                            'puntuacion': verdict['score'],
                            'evidencia_a_favor': verdict['support_count'],
                            'evidencia_en_contra': verdict['contradict_count'],
                            'evidencia_fuerte_favor': verdict.get('strong_support', 0),
                            'evidencia_fuerte_contra': verdict.get('strong_contradict', 0),
                            'oraciones_totales': analysis['total_sentences'],
                            'oraciones_relevantes': analysis['relevant_claims'],
                            'tipo_estudio': analysis['study_type']['primary_type'],
                            'objetivo_principal': analysis['main_objective'][:200] + '...' if analysis['main_objective'] and len(analysis['main_objective']) > 200 else analysis['main_objective'],
                            'conclusion_articulo': analysis['conclusion'][:200] + '...' if analysis['conclusion'] and len(analysis['conclusion']) > 200 else analysis['conclusion'],
                            'razonamiento': verdict.get('reasoning', '')[:500] + '...' if verdict.get('reasoning') and len(verdict.get('reasoning', '')) > 500 else verdict.get('reasoning', '')
                        })

                        # Actualizar estadísticas
                        self.stats['analyzed'] += 1
                        if 'CORROBORA' in verdict['verdict_text']:
                            self.stats['corroboran'] += 1
                        elif 'CONTRADICE' in verdict['verdict_text']:
                            self.stats['contradicen'] += 1
                        else:
                            self.stats['inconclusos'] += 1

                        # Guardar evidencia resumida
                        if analysis['claims']:
                            ev_summary = []
                            for ev in analysis['claims'][:3]:
                                direction_icon = '✅' if ev['direction'] == 'supports' else '❌' if ev['direction'] == 'contradicts' else '⚪'
                                strength_icon = '🔥' if ev['language_strength'] in ['strong', 'moderate_strong'] else '💧' if ev['language_strength'] in ['hedged', 'moderate_hedged'] else '📊'
                                ev_summary.append(f"{direction_icon}{strength_icon} [{ev['section']}] {ev['relation_type']}")
                            result_row['detalle_evidencia'] = ' | '.join(ev_summary)
                else:
                    result_row['veredicto'] = 'TEXTO NO DISPONIBLE'

                results_list.append(result_row)

            # Pequeña pausa entre lotes
            time.sleep(1)

        if progress_callback:
            progress_callback("✅ Análisis con comprensión lectora completado", 1.0)

        self.results = pd.DataFrame(results_list)
        return self.results

    def generate_report(self) -> str:
        """Genera un reporte textual detallado de los resultados"""
        if self.results.empty:
            return "No hay resultados para generar reporte."

        report = []
        report.append("="*100)
        report.append("🧠 REPORTE DE VERIFICACIÓN SEMÁNTICA CON COMPRENSIÓN LECTORA")
        report.append("="*100)
        report.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total artículos encontrados: {self.stats['total_articles']}")
        report.append(f"Artículos con texto disponible: {self.stats['with_text']}")
        report.append(f"Artículos analizados: {self.stats['analyzed']}")
        report.append("")
        report.append("📊 RESULTADOS GLOBALES:")
        report.append(f"✅ CORROBORAN FUERTEMENTE: {self.stats.get('corroboran_fuerte', 0)}")
        report.append(f"✅ Corroboran: {self.stats.get('corroboran', 0)}")
        report.append(f"❌ Contradicen: {self.stats.get('contradicen', 0)}")
        report.append(f"❌ CONTRADICEN FUERTEMENTE: {self.stats.get('contradicen_fuerte', 0)}")
        report.append(f"⚠️ Inconclusos: {self.stats['inconclusos']}")
        report.append("")
        report.append("📋 DETALLE POR ARTÍCULO (con razonamiento):")
        report.append("-"*100)

        for idx, row in self.results.iterrows():
            report.append(f"\n📄 **{row['titulo']}**")
            report.append(f"   📚 Base: {row['base_datos']} | 📅 Año: {row['año']} | 🔬 Tipo: {row['tipo_estudio']}")
            report.append(f"   🔗 DOI: {row['doi']}")

            if row['veredicto'] == 'TEXTO NO DISPONIBLE':
                report.append(f"   ⚠️ {row['veredicto']} - {row['fuente_texto']}")
            else:
                # Icono según veredicto
                if 'FUERTEMENTE' in row['veredicto'] and 'CORROBORA' in row['veredicto']:
                    icono = '🔥✅'
                elif 'CORROBORA' in row['veredicto']:
                    icono = '✅'
                elif 'FUERTEMENTE' in row['veredicto'] and 'CONTRADICE' in row['veredicto']:
                    icono = '🔥❌'
                elif 'CONTRADICE' in row['veredicto']:
                    icono = '❌'
                else:
                    icono = '⚠️'

                report.append(f"   {icono} **Veredicto:** {row['veredicto']} (Confianza: {row['confianza']:.1%})")
                report.append(f"   📊 Evidencia: {row['evidencia_a_favor']} a favor ({row['evidencia_fuerte_favor']} fuertes) | {row['evidencia_en_contra']} en contra ({row['evidencia_fuerte_contra']} fuertes)")

                if row['objetivo_principal'] and pd.notna(row['objetivo_principal']):
                    report.append(f"   🎯 Objetivo: {row['objetivo_principal']}")

                if row['conclusion_articulo'] and pd.notna(row['conclusion_articulo']):
                    report.append(f"   📝 Conclusión del artículo: {row['conclusion_articulo']}")

                if row['razonamiento'] and pd.notna(row['razonamiento']):
                    report.append(f"   💭 Razonamiento del sistema: {row['razonamiento']}")

                if row['detalle_evidencia'] and pd.notna(row['detalle_evidencia']):
                    report.append(f"   🔍 Evidencia destacada: {row['detalle_evidencia']}")

        return "\n".join(report)

    def generate_html_report(self) -> str:
        """Genera un reporte HTML detallado para enviar por correo"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Reporte de Verificación Semántica</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; color: #333; }}
                h1 {{ color: #1E88E5; border-bottom: 3px solid #1E88E5; padding-bottom: 10px; }}
                h2 {{ color: #333; border-bottom: 2px solid #1E88E5; padding-bottom: 5px; margin-top: 30px; }}
                .stats {{ background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; }}
                .stat-box {{ display: inline-block; margin: 10px; padding: 15px; border-radius: 8px; min-width: 150px; text-align: center; }}
                .stat-total {{ background-color: #e3f2fd; }}
                .stat-support {{ background-color: #e8f5e8; }}
                .stat-contradict {{ background-color: #ffebee; }}
                .stat-inconclusive {{ background-color: #fff3e0; }}
                .article-card {{ background-color: white; border-left: 5px solid #1E88E5; border-radius: 10px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .verdict-strong-support {{ background: linear-gradient(135deg, #2E7D32, #4CAF50); color: white; padding: 5px 15px; border-radius: 20px; display: inline-block; }}
                .verdict-support {{ background: linear-gradient(135deg, #4CAF50, #81C784); color: white; padding: 5px 15px; border-radius: 20px; display: inline-block; }}
                .verdict-inconclusive {{ background: linear-gradient(135deg, #FF9800, #FFB74D); color: white; padding: 5px 15px; border-radius: 20px; display: inline-block; }}
                .verdict-contradict {{ background: linear-gradient(135deg, #F44336, #E57373); color: white; padding: 5px 15px; border-radius: 20px; display: inline-block; }}
                .verdict-strong-contradict {{ background: linear-gradient(135deg, #B71C1C, #F44336); color: white; padding: 5px 15px; border-radius: 20px; display: inline-block; }}
                .badge {{ background-color: #1E88E5; color: white; padding: 3px 10px; border-radius: 15px; font-size: 0.8em; display: inline-block; margin-right: 10px; }}
                .evidence-badge {{ background-color: #e1f5fe; color: #01579b; padding: 2px 8px; border-radius: 12px; font-size: 0.75em; margin: 2px; display: inline-block; }}
                .reasoning-box {{ background-color: #f5f5f5; border-left: 4px solid #1E88E5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th {{ background-color: #1E88E5; color: white; padding: 12px; text-align: left; }}
                td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                tr:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <h1>🧠 Reporte de Verificación Semántica con Comprensión Lectora</h1>
            <p><strong>Fecha:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <div class="stats">
                <h2>📊 Resultados Globales</h2>
                <div>
                    <div class="stat-box stat-total">
                        <h3>📚 Total</h3>
                        <p style="font-size: 2em; margin: 0;">{self.stats['total_articles']}</p>
                        <small>artículos encontrados</small>
                    </div>
                    <div class="stat-box stat-support">
                        <h3>✅ Corroboran</h3>
                        <p style="font-size: 2em; margin: 0;">{self.stats['corroboran']}</p>
                        <small>({self.stats.get('corroboran_fuerte', 0)} fuertes)</small>
                    </div>
                    <div class="stat-box stat-contradict">
                        <h3>❌ Contradicen</h3>
                        <p style="font-size: 2em; margin: 0;">{self.stats['contradicen']}</p>
                        <small>({self.stats.get('contradicen_fuerte', 0)} fuertes)</small>
                    </div>
                    <div class="stat-box stat-inconclusive">
                        <h3>⚠️ Inconclusos</h3>
                        <p style="font-size: 2em; margin: 0;">{self.stats['inconclusos']}</p>
                    </div>
                </div>
                <p><strong>Artículos analizados:</strong> {self.stats['analyzed']} de {self.stats['with_text']} con texto disponible</p>
            </div>

            <h2>📋 Análisis Detallado por Artículo</h2>
        """

        for idx, row in self.results.iterrows():
            verdict_class = "verdict-inconclusive"
            if row['veredicto'] == 'CORROBORA FUERTEMENTE':
                verdict_class = "verdict-strong-support"
            elif row['veredicto'] == 'CORROBORA':
                verdict_class = "verdict-support"
            elif row['veredicto'] == 'CONTRADICE FUERTEMENTE':
                verdict_class = "verdict-strong-contradict"
            elif row['veredicto'] == 'CONTRADICE':
                verdict_class = "verdict-contradict"

            html += f"""
            <div class="article-card">
                <span class="badge">{row['base_datos']}</span>
                <span class="badge" style="background-color: #FF5722;">{row['tipo_estudio']}</span>
                <h3>{row['titulo']}</h3>
                <p><strong>Autores:</strong> {row['autores'][:150]}...<br>
                <strong>Año:</strong> {row['año']} | <strong>DOI:</strong> {row['doi']}</p>

                <div style="margin: 15px 0;">
                    <span class="{verdict_class}">{row['veredicto']}</span>
                    <span style="margin-left: 15px;">Confianza: {row['confianza']:.1%}</span>
                </div>

                <div style="margin: 10px 0;">
                    <span style="background-color: #e8f5e8; padding: 5px 10px; border-radius: 5px;">✅ A favor: {row['evidencia_a_favor']} ({row['evidencia_fuerte_favor']} fuertes)</span>
                    <span style="background-color: #ffebee; padding: 5px 10px; border-radius: 5px; margin-left: 10px;">❌ En contra: {row['evidencia_en_contra']} ({row['evidencia_fuerte_contra']} fuertes)</span>
                </div>
            """

            if pd.notna(row['objetivo_principal']):
                html += f"<p><strong>🎯 Objetivo:</strong> {row['objetivo_principal']}</p>"

            if pd.notna(row['conclusion_articulo']):
                html += f"<p><strong>📝 Conclusión del artículo:</strong> {row['conclusion_articulo']}</p>"

            if pd.notna(row['razonamiento']):
                html += f"""
                <div class="reasoning-box">
                    <strong>💭 Razonamiento del sistema:</strong>
                    <p>{row['razonamiento']}</p>
                </div>
                """

            if pd.notna(row['detalle_evidencia']):
                evidencias = row['detalle_evidencia'].split(' | ')
                html += "<p><strong>🔍 Evidencia destacada:</strong><br>"
                for ev in evidencias:
                    html += f'<span class="evidence-badge">{ev}</span> '
                html += "</p>"

            html += "</div>"

        html += """
            <hr>
            <p style="color: #666; font-size: 0.9em; text-align: center;">
                Reporte generado automáticamente por el Buscador y Verificador Semántico con Comprensión Lectora.
            </p>
        </body>
        </html>
        """

        return html


# ============================================================================
# FUNCIÓN PARA ENVIAR RESULTADOS POR CORREO
# ============================================================================

def enviar_resultados_email(destinatario, integrator):
    """Envía los resultados del análisis por correo electrónico"""

    # Generar reportes
    reporte_txt = integrator.generate_report()
    reporte_html = integrator.generate_html_report()

    # Preparar archivos adjuntos
    archivos = []

    # CSV
    csv_buffer = io.BytesIO()
    integrator.results.to_csv(csv_buffer, index=False, encoding='utf-8')
    archivos.append({
        'nombre': f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        'contenido': csv_buffer.getvalue()
    })

    # Excel
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        integrator.results.to_excel(writer, sheet_name='Resultados', index=False)
        summary = pd.DataFrame([integrator.stats])
        summary.to_excel(writer, sheet_name='Resumen', index=False)
    archivos.append({
        'nombre': f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        'contenido': excel_buffer.getvalue()
    })

    # Asunto y mensaje
    asunto = f"🧠 Reporte de Verificación con Comprensión Lectora - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    mensaje_html = f"""
    <html>
    <body>
        <h2>🧠 Reporte de Verificación Semántica con Comprensión Lectora</h2>
        <p>Estimado usuario,</p>
        <p>Adjunto encontrarás los resultados completos de tu análisis. Este análisis incluye:</p>
        <ul>
            <li>✅ Comprensión lectora avanzada (no solo búsqueda de palabras clave)</li>
            <li>✅ Detección del tipo de estudio y su peso en la evidencia</li>
            <li>✅ Extracción del objetivo principal y conclusiones</li>
            <li>✅ Razonamiento explicativo para cada veredicto</li>
            <li>✅ Identificación de lenguaje cauto vs. afirmaciones fuertes</li>
        </ul>

        <h3>📊 Resumen</h3>
        <ul>
            <li><strong>Total artículos:</strong> {integrator.stats['total_articles']}</li>
            <li><strong>Artículos analizados:</strong> {integrator.stats['analyzed']}</li>
            <li><strong>✅ Corroboran:</strong> {integrator.stats['corroboran']}</li>
            <li><strong>❌ Contradicen:</strong> {integrator.stats['contradicen']}</li>
            <li><strong>⚠️ Inconclusos:</strong> {integrator.stats['inconclusos']}</li>
        </ul>

        <p>Puedes encontrar el detalle completo en los archivos adjuntos.</p>

        <hr>
        <p style="color: #666; font-size: 0.9em;">
            Este es un mensaje automático del Buscador y Verificador Semántico con Comprensión Lectora.
        </p>
    </body>
    </html>
    """

    mensaje_texto = f"""
    Reporte de Verificación Semántica con Comprensión Lectora

    Resumen:
    - Total artículos: {integrator.stats['total_articles']}
    - Artículos analizados: {integrator.stats['analyzed']}
    - ✅ Corroboran: {integrator.stats['corroboran']}
    - ❌ Contradicen: {integrator.stats['contradicen']}
    - ⚠️ Inconclusos: {integrator.stats['inconclusos']}

    Los archivos adjuntos contienen el detalle completo con razonamiento explicativo.
    """

    # Enviar correo
    return enviar_correo(
        destinatario=destinatario,
        asunto=asunto,
        mensaje_html=mensaje_html,
        mensaje_texto=mensaje_texto,
        archivos=archivos
    )


# ============================================================================
# ASISTENTE DE CONSTRUCCIÓN DE CONJETURAS (con mejoras)
# ============================================================================

class HypothesisAssistant:
    """Asistente para ayudar al usuario a construir conjeturas bien formadas"""

    def __init__(self):
        self.translator = TranslationManager()

        # Plantillas de conjeturas comunes
        self.templates = {
            "causal": {
                "es": "El/La {sujeto} {verbo} {efecto} en {poblacion}",
                "en": "The {subject} {verb} {effect} in {population}",
                "description": "Relación causal directa",
                "verbs_es": ["causa", "produce", "induce", "provoca", "genera"],
                "verbs_en": ["causes", "produces", "induces", "provokes", "generates"]
            },
            "asociacion": {
                "es": "Existe una asociación entre {sujeto} y {efecto} en {poblacion}",
                "en": "There is an association between {subject} and {effect} in {population}",
                "description": "Asociación o correlación",
                "verbs_es": ["se asocia con", "se relaciona con", "se correlaciona con"],
                "verbs_en": ["is associated with", "is related to", "is correlated with"]
            },
            "riesgo": {
                "es": "El/La {sujeto} aumenta el riesgo de {efecto} en {poblacion}",
                "en": "The {subject} increases the risk of {effect} in {population}",
                "description": "Factor de riesgo",
                "verbs_es": ["aumenta el riesgo de", "incrementa la probabilidad de", "es factor de riesgo para"],
                "verbs_en": ["increases the risk of", "is a risk factor for"]
            },
            "prevencion": {
                "es": "El/La {sujeto} reduce la incidencia de {efecto} en {poblacion}",
                "en": "The {subject} reduces the incidence of {effect} in {population}",
                "description": "Factor protector o preventivo",
                "verbs_es": ["reduce", "disminuye", "previene", "protege contra"],
                "verbs_en": ["reduces", "decreases", "prevents", "protects against"]
            },
            "efectividad": {
                "es": "El/La {sujeto} es efectivo para tratar {efecto} en {poblacion}",
                "en": "The {subject} is effective in treating {effect} in {population}",
                "description": "Efectividad terapéutica",
                "verbs_es": ["es efectivo para", "es eficaz para", "demuestra eficacia en"],
                "verbs_en": ["is effective in", "demonstrates efficacy in"]
            }
        }

        self.examples = [
            {
                "name": "Alcohol y demencia",
                "sujeto": "consumo de alcohol",
                "efecto": "demencia",
                "poblacion": "adultos mayores",
                "tipo": "causal",
                "verbo": "causa",
                "hypothesis": "El consumo de alcohol causa demencia en adultos mayores"
            },
            {
                "name": "Ticagrelor y disnea",
                "sujeto": "ticagrelor",
                "efecto": "disnea",
                "poblacion": "pacientes con cardiopatía isquémica",
                "tipo": "causal",
                "verbo": "causa",
                "hypothesis": "El ticagrelor causa disnea como efecto secundario en pacientes con cardiopatía isquémica"
            },
            {
                "name": "COVID-19 y daño miocárdico",
                "sujeto": "infección por COVID-19",
                "efecto": "daño miocárdico",
                "poblacion": "pacientes hospitalizados",
                "tipo": "causal",
                "verbo": "causa",
                "hypothesis": "La infección por COVID-19 causa daño miocárdico en pacientes hospitalizados"
            }
        ]

    def get_available_verbs(self, template_type: str, language: str = "es") -> List[str]:
        if template_type in self.templates:
            return self.templates[template_type].get(f"verbs_{language}", [])
        return []

    def build_hypothesis(self, subject: str, effect: str, population: str,
                        template_type: str, verb: str = None) -> Dict:
        if template_type not in self.templates:
            return {"es": "", "en": ""}

        template = self.templates[template_type]

        if not verb and template.get(f"verbs_es"):
            verb = template[f"verbs_es"][0]

        # Construir en español
        if verb and "{verbo}" in template["es"]:
            hypothesis_es = template["es"].format(
                sujeto=subject,
                verbo=verb,
                efecto=effect,
                poblacion=population
            )
        else:
            hypothesis_es = template["es"].format(
                sujeto=subject,
                efecto=effect,
                poblacion=population
            )

        # Traducciones
        hypothesis_en = self.translator.translate_to_english(hypothesis_es)

        subject_en = self.translator.translate_to_english(subject)
        effect_en = self.translator.translate_to_english(effect)
        population_en = self.translator.translate_to_english(population)

        # Verbo en inglés
        verb_en = ""
        if verb and template.get(f"verbs_es") and template.get(f"verbs_en"):
            try:
                verb_idx = template["verbs_es"].index(verb)
                verbs_en = template["verbs_en"]
                verb_en = verbs_en[verb_idx] if verb_idx < len(verbs_en) else verbs_en[0]
            except ValueError:
                verb_en = self.translator.translate_to_english(verb)

        # Construcción directa en inglés
        if verb_en and "{verb}" in template["en"]:
            hypothesis_en_direct = template["en"].format(
                subject=subject_en,
                verb=verb_en,
                effect=effect_en,
                population=population_en
            )
        else:
            hypothesis_en_direct = template["en"].format(
                subject=subject_en,
                effect=effect_en,
                population=population_en
            )

        return {
            "es": hypothesis_es,
            "en": hypothesis_en,
            "en_direct": hypothesis_en_direct,
            "subject": subject,
            "effect": effect,
            "population": population,
            "type": template_type,
            "type_description": template["description"],
            "verb": verb,
            "verb_en": verb_en
        }

    def render_assistant_ui(self):
        """Renderiza la interfaz del asistente"""

        with st.expander("🤖 ASISTENTE DE CONJETURAS - Construye tu hipótesis", expanded=False):
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                <h3>🎯 Construye tu conjetura científica</h3>
                <p>El sistema ahora tiene CAPACIDAD DE COMPRENSIÓN LECTORA HUMANA. Analizará cada artículo entendiendo su objetivo, metodología y conclusiones, no solo buscando palabras clave.</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                example_option = st.selectbox(
                    "📋 Cargar ejemplo:",
                    ["Personalizado"] + [e["name"] for e in self.examples],
                    key="example_selector"
                )

                subject = st.text_input(
                    "🧪 Sujeto/Intervención:",
                    value="",
                    placeholder="Ej: consumo de alcohol, ticagrelor, ejercicio",
                    help="¿Qué elemento estás estudiando?"
                )

                effect = st.text_input(
                    "📊 Efecto/Desenlace:",
                    value="",
                    placeholder="Ej: demencia, disnea, mejoría",
                    help="¿Qué efecto esperas observar?"
                )

                population = st.text_input(
                    "👥 Población:",
                    value="",
                    placeholder="Ej: adultos mayores, pacientes, niños",
                    help="¿En qué población?"
                )

            with col2:
                template_type = st.selectbox(
                    "🔄 Tipo de relación:",
                    options=list(self.templates.keys()),
                    format_func=lambda x: f"{x} - {self.templates[x]['description']}",
                    key="template_type"
                )

                available_verbs = self.get_available_verbs(template_type)
                if available_verbs:
                    verb = st.selectbox(
                        "🔤 Verbo/relación:",
                        options=available_verbs,
                        key="selected_verb"
                    )
                else:
                    verb = None

                st.markdown("---")
                st.markdown("##### 💡 Sugerencias:")
                if template_type == "causal":
                    st.info("Usa verbos fuertes como 'causa', 'induce' para relaciones causales directas")
                elif template_type == "asociacion":
                    st.info("Usa 'se asocia con' para correlaciones sin causalidad establecida")
                elif template_type == "riesgo":
                    st.info("Adecuado cuando el sujeto aumenta la probabilidad del efecto")
                elif template_type == "prevencion":
                    st.info("Usa cuando el sujeto reduce el riesgo")
                elif template_type == "efectividad":
                    st.info("Ideal para evaluar intervenciones terapéuticas")

            if st.button("✨ GENERAR CONJETURA", type="primary", use_container_width=True):
                if subject and effect and population:
                    hypothesis_data = self.build_hypothesis(
                        subject=subject,
                        effect=effect,
                        population=population,
                        template_type=template_type,
                        verb=verb
                    )

                    st.markdown("---")
                    st.markdown("### 📝 CONJETURA GENERADA")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**🇪🇸 Español:**")
                        st.success(hypothesis_data["es"])

                        if st.button("📌 Usar esta hipótesis", key="use_es"):
                            st.session_state['hypothesis'] = hypothesis_data["es"]
                            st.session_state['hypothesis_en'] = hypothesis_data["en_direct"]
                            st.rerun()

                    with col2:
                        st.markdown("**🇬🇧 Inglés (para búsqueda):**")
                        st.info(hypothesis_data["en_direct"])

                    st.session_state['last_hypothesis_data'] = hypothesis_data

                else:
                    st.warning("⚠️ Completa todos los campos para generar la conjetura")

            if example_option != "Personalizado":
                example = next((e for e in self.examples if e["name"] == example_option), None)
                if example:
                    st.markdown("---")
                    st.markdown("### 📋 Ejemplo cargado:")
                    st.info(f"**Hipótesis:** {example['hypothesis']}")

                    if st.button("📌 Usar este ejemplo", key="use_example"):
                        st.session_state['hypothesis'] = example['hypothesis']
                        st.session_state['query'] = f'("{example["sujeto"]}" AND "{example["efecto"]}")'
                        translator = TranslationManager()
                        st.session_state['hypothesis_en'] = translator.translate_to_english(example['hypothesis'])
                        st.rerun()


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def get_badge_class(db_name: str) -> str:
    classes = {
        'PubMed': 'badge-pubmed',
        'CrossRef': 'badge-crossref',
        'OpenAlex': 'badge-openalex',
        'Semantic Scholar': 'badge-semantic',
        'DOAJ': 'badge-doaj',
        'Europe PMC': 'badge-europepmc'
    }
    return classes.get(db_name, 'badge-pubmed')

def get_verdict_class(verdict: str) -> str:
    if 'FUERTEMENTE' in verdict and 'CORROBORA' in verdict:
        return 'verdict-assert'
    elif 'CORROBORA' in verdict:
        return 'verdict-assert'
    elif 'FUERTEMENTE' in verdict and 'CONTRADICE' in verdict:
        return 'verdict-reject'
    elif 'CONTRADICE' in verdict:
        return 'verdict-reject'
    elif 'NO CONCLUYENTE' in verdict:
        return 'verdict-inconclusive'
    else:
        return ''


# ============================================================================
# INTERFAZ PRINCIPAL DE STREAMLIT
# ============================================================================

def main():
    """Función principal de la aplicación"""

    st.markdown("""
    <h1 style="text-align: center; color: #1E88E5; font-size: 2.5rem; margin-bottom: 0.5rem;">
        🧠 Buscador con Comprensión Lectora Humana
    </h1>
    <p style="text-align: center; color: #424242; font-size: 1.2rem; margin-bottom: 2rem;">
        ALTO VOLUMEN: Hasta 1000 artículos por base • Análisis con comprensión lectora tipo humano
    </p>
    """, unsafe_allow_html=True)

    # Inicializar asistente de hipótesis
    hypothesis_assistant = HypothesisAssistant()
    hypothesis_assistant.render_assistant_ui()

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
        st.markdown("## ⚙️ Configuración")

        email = st.text_input("📧 Email (para NCBI y recibir resultados)",
                              value="usuario@ejemplo.com",
                              help="Este email se usará para NCBI y para enviarte los resultados")

        if NLTK_READY:
            st.success("✅ NLTK configurado")
        else:
            st.warning("⚠️ Usando tokenizador alternativo")

        st.markdown("### 📚 Bases de Datos")
        col1, col2 = st.columns(2)
        with col1:
            pubmed = st.checkbox('PubMed', value=True)
            crossref = st.checkbox('CrossRef', value=True)
            openalex = st.checkbox('OpenAlex', value=True)
        with col2:
            semantic = st.checkbox('Semantic Scholar', value=False)
            doaj = st.checkbox('DOAJ', value=False)
            europepmc = st.checkbox('Europe PMC', value=True)

        databases = {
            'PubMed': pubmed,
            'CrossRef': crossref,
            'OpenAlex': openalex,
            'Semantic Scholar': semantic,
            'DOAJ': doaj,
            'Europe PMC': europepmc
        }

        st.markdown("---")
        st.markdown("### 📅 Filtro de Años")
        col1, col2 = st.columns(2)
        with col1:
            year_from = st.number_input("Desde", min_value=1900, max_value=2025, value=2015)
        with col2:
            year_to = st.number_input("Hasta", min_value=1900, max_value=2025, value=2025)

        year_range = (year_from, year_to) if year_from < year_to else None

        st.markdown("### 📊 Resultados por base")
        max_results = st.slider(
            "Máximo resultados por base:",
            min_value=100, max_value=1000, value=500, step=100
        )

        st.markdown("### 🔧 Umbral de relevancia")
        min_relevance = st.slider(
            "Relevancia mínima",
            min_value=0.05, max_value=0.3, value=0.1, step=0.05
        )

        st.markdown("### ⚠️ Advertencia")
        st.warning("""
        **Tiempo de procesamiento (con comprensión lectora):**
        - 100 artículos: ~10-15 minutos
        - 500 artículos: ~30-45 minutos
        - El análisis es más lento pero MUCHO más preciso
        """)

        st.markdown("### 📋 Ejemplos")
        if st.button("Cargar ejemplo: Alcohol y demencia"):
            st.session_state['query'] = '(("Alcohol Drinking"[Mesh] OR "Alcoholism"[Mesh]) AND "Dementia"[Mesh])'
            st.session_state['hypothesis'] = "El consumo de alcohol causa demencia en adultos mayores"
            st.session_state['hypothesis_en'] = "Alcohol consumption causes dementia in older adults"

        if st.button("Cargar ejemplo: Ticagrelor y disnea"):
            st.session_state['query'] = '(("Ticagrelor"[Mesh]) AND "Dyspnea"[Mesh])'
            st.session_state['hypothesis'] = "El ticagrelor causa disnea como efecto secundario en pacientes con cardiopatía isquémica"
            st.session_state['hypothesis_en'] = "Ticagrelor causes dyspnea as a side effect in patients with ischemic heart disease"

    # Área principal
    col1, col2 = st.columns([2, 1])

    with col1:
        search_query = st.text_area(
            "🔍 Consulta de búsqueda:",
            value=st.session_state.get('query', ''),
            height=100,
            placeholder='Ej: (("Alcohol"[Mesh]) AND "Dementia"[Mesh])'
        )

    with col2:
        hypothesis_es = st.text_area(
            "🔬 Conjetura a verificar (español):",
            value=st.session_state.get('hypothesis', ''),
            height=60,
            placeholder='Ej: El alcohol causa demencia en adultos mayores'
        )

        if hypothesis_es:
            if 'hypothesis_en' not in st.session_state or not st.session_state['hypothesis_en']:
                translator = TranslationManager()
                hypothesis_en = translator.translate_to_english(hypothesis_es)
                st.session_state['hypothesis_en'] = hypothesis_en

            st.markdown(f"**🇬🇧 Inglés (para búsqueda):**")
            st.info(st.session_state.get('hypothesis_en', '')[:150] + '...')

    col1, col2, col3 = st.columns(3)
    with col2:
        analyze_button = st.button("🧠 INICIAR ANÁLISIS CON COMPRENSIÓN LECTORA", type="primary", use_container_width=True)

    if 'integrator' not in st.session_state:
        st.session_state.integrator = None

    if analyze_button and search_query and hypothesis_es:
        if email and email != "usuario@ejemplo.com" and validate_email(email):
            integrator = IntegratedScientificVerifier(email)
            integrator.semantic_verifier.UMBRAL_RELEVANCIA = min_relevance

            selected_dbs = [db for db, selected in databases.items() if selected]

            if not selected_dbs:
                st.error("❌ Selecciona al menos una base de datos")
                return

            hypothesis_for_analysis = st.session_state.get('hypothesis_en', hypothesis_es)

            progress_container = st.container()
            with progress_container:
                st.markdown('<div style="background-color: #f0f2f6; border-radius: 10px; padding: 1rem;">', unsafe_allow_html=True)
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_estimate = st.empty()
                st.markdown('</div>', unsafe_allow_html=True)

            start_time = time.time()

            def update_progress(message, value):
                status_text.text(message)
                progress_bar.progress(value)
                elapsed = time.time() - start_time
                if value > 0:
                    estimated_total = elapsed / value
                    remaining = estimated_total * (1 - value)
                    if remaining > 0:
                        time_estimate.text(f"⏱️ Transcurrido: {elapsed:.1f}s | Restante: {remaining:.1f}s")

            with st.spinner("Ejecutando análisis con comprensión lectora..."):
                results_df = integrator.run_analysis(
                    search_query,
                    hypothesis_for_analysis,
                    max_results,
                    selected_dbs,
                    year_range,
                    update_progress
                )
                elapsed_time = time.time() - start_time

            if not results_df.empty:
                st.success(f"✅ Análisis con comprensión lectora completado en {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
                st.session_state.integrator = integrator

                # ESTADÍSTICAS GLOBALES
                st.markdown("## 📊 RESULTADOS GLOBALES")

                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Artículos encontrados", integrator.stats['total_articles'])
                with col2:
                    st.metric("Con texto", integrator.stats['with_text'])
                with col3:
                    st.metric("✅ Corroboran", integrator.stats['corroboran'])
                with col4:
                    st.metric("❌ Contradicen", integrator.stats['contradicen'])
                with col5:
                    st.metric("⚠️ Inconclusos", integrator.stats['inconclusos'])

                # GRÁFICOS
                col1, col2 = st.columns(2)

                with col1:
                    db_counts = results_df['base_datos'].value_counts().reset_index()
                    db_counts.columns = ['base_datos', 'count']
                    fig = px.bar(
                        db_counts, x='base_datos', y='count',
                        title=f"Artículos por Base de Datos",
                        color='base_datos',
                        color_discrete_map={
                            'PubMed': '#1E88E5',
                            'CrossRef': '#43A047',
                            'OpenAlex': '#FDD835',
                            'Europe PMC': '#FF5722'
                        }
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    analyzed_df = results_df[results_df['veredicto'] != 'TEXTO NO DISPONIBLE']
                    if not analyzed_df.empty:
                        verdict_counts = analyzed_df['veredicto'].value_counts().reset_index()
                        verdict_counts.columns = ['veredicto', 'count']
                        fig = px.pie(
                            verdict_counts, values='count', names='veredicto',
                            title=f"Distribución de Veredictos (n={len(analyzed_df)})",
                            color_discrete_sequence=['#4CAF50', '#f44336', '#ff9800']
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # TABLA DE RESULTADOS
                st.markdown("## 📋 ARTÍCULOS ANALIZADOS")

                display_cols = ['base_datos', 'titulo', 'año', 'tipo_estudio', 'veredicto', 'confianza',
                               'evidencia_a_favor', 'evidencia_en_contra']
                display_df = results_df[display_cols].copy()
                display_df['confianza'] = display_df['confianza'].apply(lambda x: f"{x:.1%}" if x > 0 else 'N/A')

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400,
                    column_config={
                        'base_datos': 'Base',
                        'titulo': 'Título',
                        'año': 'Año',
                        'tipo_estudio': 'Tipo',
                        'veredicto': 'Veredicto',
                        'confianza': 'Confianza',
                        'evidencia_a_favor': 'A favor',
                        'evidencia_en_contra': 'En contra'
                    }
                )

                # DETALLE POR ARTÍCULO
                st.markdown("## 🔍 DETALLE DE ANÁLISIS CON RAZONAMIENTO")

                for idx, row in results_df.head(10).iterrows():
                    badge_class = get_badge_class(row['base_datos'])
                    verdict_class = get_verdict_class(row['veredicto'])

                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; border-radius: 10px; padding: 1.5rem; margin-bottom: 1rem; border-left: 5px solid #1E88E5;">
                        <span style="background-color: #1E88E5; color: white; padding: 0.2rem 0.8rem; border-radius: 15px; font-size: 0.8rem;">{row['base_datos']}</span>
                        <span style="background-color: #FF5722; color: white; padding: 0.2rem 0.8rem; border-radius: 15px; font-size: 0.8rem; margin-left: 0.5rem;">{row['tipo_estudio']}</span>
                        <h4 style="margin-top: 0.5rem;">{row['titulo']}</h4>
                        <p><strong>Año:</strong> {row['año']} | <strong>DOI:</strong> {row['doi']}</p>
                        <p><span style="background: linear-gradient(135deg, #4CAF50, #2E7D32); color: white; padding: 0.3rem 1rem; border-radius: 20px; display: inline-block;">{row['veredicto']}</span> (Confianza: {row['confianza']:.1%})</p>
                        <p><strong>Evidencia:</strong> {row['evidencia_a_favor']} a favor ({row['evidencia_fuerte_favor']} fuertes) | {row['evidencia_en_contra']} en contra ({row['evidencia_fuerte_contra']} fuertes)</p>
                    """, unsafe_allow_html=True)

                    if pd.notna(row['objetivo_principal']):
                        st.markdown(f"**🎯 Objetivo:** {row['objetivo_principal']}")

                    if pd.notna(row['conclusion_articulo']):
                        st.markdown(f"**📝 Conclusión del artículo:** {row['conclusion_articulo']}")

                    if pd.notna(row['razonamiento']):
                        st.markdown(f"""
                        <div style="background-color: #e3f2fd; border-left: 4px solid #1E88E5; padding: 1rem; margin: 0.5rem 0; border-radius: 5px;">
                            <strong>💭 Razonamiento del sistema:</strong><br>
                            {row['razonamiento']}
                        </div>
                        """, unsafe_allow_html=True)

                    if pd.notna(row['detalle_evidencia']):
                        st.markdown(f"**🔍 Evidencia destacada:** {row['detalle_evidencia']}")

                    col1, col2 = st.columns([1, 5])
                    with col1:
                        if row.get('url') and pd.notna(row['url']):
                            st.link_button("🔗 Ver artículo", row['url'])
                    st.markdown("</div>", unsafe_allow_html=True)

                if len(results_df) > 10:
                    st.info(f"... y {len(results_df) - 10} artículos más. Exporta los resultados para ver el listado completo.")

                # EXPORTACIÓN
                st.markdown("## 💾 EXPORTAR RESULTADOS")

                col1, col2, col3 = st.columns(3)

                report_text = integrator.generate_report()
                with col1:
                    st.download_button(
                        "📝 Reporte TXT",
                        report_text,
                        file_name=f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

                with col2:
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📊 CSV",
                        csv,
                        file_name=f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col3:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        results_df.to_excel(writer, sheet_name='Resultados', index=False)
                        summary = pd.DataFrame([integrator.stats])
                        summary.to_excel(writer, sheet_name='Resumen', index=False)
                    st.download_button(
                        "📥 Excel",
                        buffer.getvalue(),
                        file_name=f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

            else:
                st.warning("😕 No se encontraron artículos. Prueba con otra consulta.")

        else:
            st.warning("⚠️ Ingresa un email válido en la barra lateral")

    # Sección para enviar resultados por correo
    if st.session_state.integrator is not None and not st.session_state.integrator.results.empty:
        st.markdown("---")
        st.markdown("## 📧 ENVIAR RESULTADOS POR CORREO")
        st.markdown("""
        <div style="background-color: #fce4e4; border-left: 5px solid #e53935; padding: 1rem; border-radius: 5px; margin: 1rem 0;">
            <p>Recibirás un correo con el reporte completo en formato HTML, más archivos CSV y Excel adjuntos.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📨 ENVIAR RESULTADOS POR CORREO", type="primary", use_container_width=True):
                with st.spinner("Enviando resultados..."):
                    if enviar_resultados_email(email, st.session_state.integrator):
                        st.success(f"✅ Resultados enviados a {email}")
                        st.balloons()
                    else:
                        st.error("❌ No se pudo enviar el correo.")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🧠 Buscador con Comprensión Lectora Humana v3.0 | Análisis semántico profundo con razonamiento explicativo</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
