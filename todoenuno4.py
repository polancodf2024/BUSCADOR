"""
Buscador y Verificador Semántico Integrado - VERSIÓN COMPLETA CORREGIDA
CORRECCIONES CRÍTICAS:
1. PubMed: Manejo correcto de términos MeSH con comas
2. Límite de resultados: Ahora respeta hasta 1000 artículos por base
3. Ejemplos: Corregida la carga y consultas optimizadas
4. Persistencia: Los resultados ya NO se borran después del análisis
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
from typing import List, Dict, Tuple, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import Counter
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
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="🔬 Buscador y Verificador Semántico Integrado",
    page_icon="🔬",
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
            self.SMTP_SERVER,
            self.SMTP_PORT,
            self.EMAIL_USER,
            self.EMAIL_PASSWORD
        ])

EMAIL_CONFIG = EmailConfig()

def validate_email(email):
    if not email or email == "":
        return False
    if email == "usuario@ejemplo.com":
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def enviar_correo(destinatario, asunto, mensaje_html=None, mensaje_texto=None, archivos=None):
    if not EMAIL_CONFIG.available:
        st.error("Configuración de correo no disponible. Verifica los secrets de la aplicación.")
        return False
    
    if not validate_email(destinatario):
        st.error("Dirección de correo no válida")
        return False
    
    if not asunto or (not mensaje_html and not mensaje_texto):
        st.error("Faltan datos requeridos para enviar el correo")
        return False
    
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = EMAIL_CONFIG.EMAIL_USER
        msg['To'] = destinatario
        msg['Subject'] = asunto
        msg['Date'] = formatdate(localtime=True)
        
        if mensaje_texto:
            msg.attach(MIMEText(mensaje_texto, 'plain', 'utf-8'))
        
        if mensaje_html:
            msg.attach(MIMEText(mensaje_html, 'html', 'utf-8'))
        
        if archivos:
            for archivo in archivos:
                if len(archivo['contenido']) > EMAIL_CONFIG.MAX_FILE_SIZE_MB * 1024 * 1024:
                    st.warning(f"El archivo {archivo['nombre']} excede el tamaño máximo y no será enviado")
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
            nltk.download('punkt', quiet=True)
        except:
            st.warning("No se pudo descargar punkt. Usando tokenización alternativa.")
            return False
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except:
            st.warning("No se pudieron descargar stopwords. Usando lista básica.")
            return False
    
    try:
        from nltk.corpus import stopwords
        return True
    except:
        return False

NLTK_READY = setup_nltk()

if NLTK_READY:
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    from nltk.tokenize import sent_tokenize

# ============================================================================
# ESTILOS CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .verdict-assert {
        background: linear-gradient(135deg, #4CAF50, #2E7D32);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
    .verdict-reject {
        background: linear-gradient(135deg, #f44336, #b71c1c);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
    .verdict-inconclusive {
        background: linear-gradient(135deg, #ff9800, #bf360c);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
    .badge-pubmed { 
        background-color: #1E88E5; 
        color: white; 
        padding: 0.2rem 0.8rem; 
        border-radius: 15px; 
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .badge-crossref { 
        background-color: #43A047; 
        color: white; 
        padding: 0.2rem 0.8rem; 
        border-radius: 15px; 
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .badge-openalex { 
        background-color: #FDD835; 
        color: #333; 
        padding: 0.2rem 0.8rem; 
        border-radius: 15px; 
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .badge-europepmc { 
        background-color: #FF5722; 
        color: white; 
        padding: 0.2rem 0.8rem; 
        border-radius: 15px; 
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .evidence-box {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 0.5rem;
        margin: 0.2rem 0;
        font-size: 0.85rem;
    }
    .progress-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .email-box {
        background-color: #fce4e4;
        border-left: 5px solid #e53935;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .assistant-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .tip-box {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TOKENIZADOR ALTERNATIVO
# ============================================================================

def simple_sent_tokenize(text: str) -> List[str]:
    abbreviations = ['Dr.', 'Dra.', 'Prof.', 'vs.', 'Fig.', 'Eq.', 'et al.', 
                    'i.e.', 'e.g.', 'p.ej.', 'vol.', 'no.', 'pp.', 'eds.']
    
    for i, abbr in enumerate(abbreviations):
        text = text.replace(abbr, f"ABBR{i}")
    
    sentences = re.split(r'[.!?]+', text)
    
    for i, abbr in enumerate(abbreviations):
        sentences = [s.replace(f"ABBR{i}", abbr) for s in sentences]
    
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    
    return sentences

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
        cache_key = f"es_{text[:100]}"
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
        cache_key = f"en_{text[:100]}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        try:
            result = self.translator_en.translate(text[:3500])
            self.cache[cache_key] = result
            return result
        except:
            return text

# ============================================================================
# ASISTENTE DE CONSTRUCCIÓN DE CONJETURAS
# ============================================================================

class HypothesisAssistant:
    def __init__(self):
        self.translator = TranslationManager()
        
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
                "name": "Ticagrelor y disnea",
                "sujeto": "ticagrelor",
                "efecto": "disnea",
                "poblacion": "pacientes con cardiopatía isquémica",
                "tipo": "causal",
                "verbo": "causa",
                "hypothesis": "El ticagrelor causa disnea como efecto secundario en pacientes con cardiopatía isquémica"
            },
            {
                "name": "Ruptura cardiaca postinfarto",
                "sujeto": "ruptura cardiaca postinfarto",
                "efecto": "patrones anatómicos de ruptura",
                "poblacion": "pacientes con infarto agudo de miocardio",
                "tipo": "asociacion",
                "verbo": "sigue",
                "hypothesis": "En la ruptura cardiaca postinfarto, el corazón se rompe siguiendo patrones anatómicos reconocibles"
            },
            {
                "name": "Ejercicio y diabetes",
                "sujeto": "ejercicio físico regular",
                "efecto": "diabetes tipo 2",
                "poblacion": "adultos con sobrepeso",
                "tipo": "prevencion",
                "verbo": "reduce la incidencia de",
                "hypothesis": "El ejercicio físico regular reduce la incidencia de diabetes tipo 2 en adultos con sobrepeso"
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
        
        if verb:
            if "{verbo}" in template["es"]:
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
        else:
            hypothesis_es = template["es"].format(
                sujeto=subject,
                efecto=effect,
                poblacion=population
            )
        
        hypothesis_en = self.translator.translate_to_english(hypothesis_es)
        
        subject_en = self.translator.translate_to_english(subject)
        effect_en = self.translator.translate_to_english(effect)
        population_en = self.translator.translate_to_english(population)
        
        verb_en = ""
        if verb and template.get(f"verbs_es") and template.get(f"verbs_en"):
            try:
                verb_idx = template["verbs_es"].index(verb)
                verbs_en = template["verbs_en"]
                verb_en = verbs_en[verb_idx] if verb_idx < len(verbs_en) else verbs_en[0]
            except ValueError:
                verb_en = self.translator.translate_to_english(verb)
        
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
        with st.expander("🤖 ASISTENTE DE CONJETURAS - Ayuda a construir tu hipótesis", expanded=False):
            st.markdown('<div class="assistant-box">', unsafe_allow_html=True)
            st.markdown("### 🎯 Construye tu conjetura científica")
            st.markdown("Completa los siguientes campos para generar una hipótesis bien formada:")
            
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
                    placeholder="Ej: ticagrelor, ejercicio, vacuna, fármaco X"
                )
                
                effect = st.text_input(
                    "📊 Efecto/Desenlace:",
                    value="",
                    placeholder="Ej: disnea, mejoría, mortalidad, efecto secundario"
                )
                
                population = st.text_input(
                    "👥 Población:",
                    value="",
                    placeholder="Ej: pacientes con cardiopatía, adultos mayores, niños"
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
                    st.info("Usa 'se asocia con', 'se relaciona con' para correlaciones sin causalidad establecida")
                elif template_type == "riesgo":
                    st.info("Adecuado cuando el sujeto aumenta la probabilidad del efecto")
                elif template_type == "prevencion":
                    st.info("Usa cuando el sujeto reduce el riesgo o protege contra el efecto")
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
                        
                        if st.button("📌 Usar esta hipótesis en español", key="use_es"):
                            st.session_state['hypothesis'] = hypothesis_data["es"]
                            st.session_state['hypothesis_en'] = hypothesis_data["en_direct"]
                            st.rerun()
                    
                    with col2:
                        st.markdown("**🇬🇧 Inglés (para búsqueda):**")
                        st.info(hypothesis_data["en_direct"])
                        
                        if st.button("📌 Usar esta hipótesis (inglés)", key="use_en"):
                            st.session_state['hypothesis'] = hypothesis_data["es"]
                            st.session_state['hypothesis_en'] = hypothesis_data["en_direct"]
                            st.rerun()
                    
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
                        translator = TranslationManager()
                        st.session_state['hypothesis_en'] = translator.translate_to_english(example['hypothesis'])
                        st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="tip-box">', unsafe_allow_html=True)
            st.markdown("""
            **💡 Consejos para una buena conjetura:**
            
            1. **Sé específico**: Cuanto más específico, mejor podrá verificar la evidencia
            2. **Define claramente**: El sujeto, efecto y población deben estar claramente definidos
            3. **Usa terminología médica**: Los artículos científicos usan términos MeSH estandarizados
            4. **Considera la dirección**: ¿Es causalidad, asociación, riesgo o protección?
            5. **Población relevante**: Especifica edad, condición, contexto cuando sea relevante
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    def translate_hypothesis_for_search(self, hypothesis_es: str) -> str:
        if not hypothesis_es:
            return ""
        
        if 'hypothesis_en' in st.session_state and st.session_state['hypothesis_en']:
            return st.session_state['hypothesis_en']
        
        return self.translator.translate_to_english(hypothesis_es)

# ============================================================================
# VERIFICADOR SEMÁNTICO AVANZADO
# ============================================================================

class AdvancedSemanticVerifier:
    def __init__(self):
        if NLTK_READY:
            self.stop_words_es = set(stopwords.words('spanish'))
            self.stop_words_en = set(stopwords.words('english'))
            self.stemmer = SnowballStemmer('spanish')
        else:
            self.stop_words_es = {'el', 'la', 'los', 'las', 'de', 'del', 'y', 'o', 
                                  'a', 'en', 'por', 'para', 'con', 'sin', 'sobre'}
            self.stop_words_en = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on',
                                 'at', 'to', 'for', 'with', 'without', 'by'}
            self.stemmer = None
        
        self.translator = TranslationManager()
        
        self.section_weights = {
            'abstract': 1.5,
            'introduction': 0.6,
            'methods': 0.4,
            'results': 1.8,
            'discussion': 1.5,
            'conclusion': 1.8,
            'unknown': 1.0
        }
        
        self.section_keywords = {
            'abstract': ['abstract', 'resumen', 'background', 'objective', 'objetivo', 
                        'methods', 'métodos', 'results', 'resultados'],
            'introduction': ['introduction', 'introducción', 'background', 'antecedentes'],
            'methods': ['methods', 'métodos', 'methodology', 'metodología', 
                       'material and methods', 'material y métodos'],
            'results': ['results', 'resultados', 'findings', 'hallazgos'],
            'discussion': ['discussion', 'discusión'],
            'conclusion': ['conclusion', 'conclusiones', 'concluding', 'conclusión']
        }
        
        self.certainty_levels = {
            'definitive': {
                'terms': ['demonstrates', 'proves', 'establishes', 'conclusive', 
                         'definitively', 'undoubtedly', 'certainly'],
                'weight': 1.5
            },
            'strong': {
                'terms': ['strongly suggests', 'provides evidence', 'indicates',
                         'shows', 'reveals', 'confirms'],
                'weight': 1.3
            },
            'moderate': {
                'terms': ['suggests', 'supports', 'consistent with', 'implies',
                         'appears to', 'seems to'],
                'weight': 1.0
            },
            'weak': {
                'terms': ['may', 'might', 'could', 'possibly', 'potentially',
                         'preliminary', 'tentative', 'speculative'],
                'weight': 0.5
            }
        }
        
        self.negation_patterns = [
            r'no\s+(hay|existe|se\s+observó|se\s+encontró)\s+(asociación|relación|correlación)',
            r'no\s+(se\s+)?(demostró|evidenció|confirmó)\s+',
            r'no\s+(fue|resultó)\s+(significativo|estadísticamente\s+significativo)',
            r'p\s*[>=]\s*0\.0[5-9]',
            r'p\s*>\s*0\.05',
            r'not\s+(associated|related|correlated)',
            r'no\s+(significant|statistically\s+significant)'
        ]
        
        self.relation_types = {
            'causal': {
                'terms': ['causes', 'leads to', 'results in', 'induced by',
                         'due to', 'attributed to', 'responsible for'],
                'weight': 1.4
            },
            'association': {
                'terms': ['associated with', 'related to', 'linked to',
                         'correlated with', 'connection between'],
                'weight': 1.2
            },
            'risk': {
                'terms': ['risk factor', 'increases risk', 'higher risk',
                         'elevated risk', 'predisposes to'],
                'weight': 1.3
            },
            'protective': {
                'terms': ['protective', 'reduces risk', 'decreases risk',
                         'prevents', 'lowers risk'],
                'weight': 1.1
            }
        }
        
        self.study_type_weights = {
            'meta-analysis': 2.5,
            'systematic review': 2.3,
            'randomized controlled trial': 2.0,
            'cohort study': 1.6,
            'case-control study': 1.4,
            'cross-sectional study': 1.2,
            'case series': 0.8,
            'case report': 0.5,
            'editorial': 0.3,
            'letter': 0.2,
            'comment': 0.2
        }
        
        self.study_type_patterns = {
            'meta-analysis': ['meta-analysis', 'meta analysis', 'metaanalysis'],
            'systematic review': ['systematic review', 'systematic literature review'],
            'randomized controlled trial': ['randomized controlled trial', 'randomised controlled trial', 'rct'],
            'cohort study': ['cohort study', 'cohort analysis', 'prospective cohort', 'retrospective cohort'],
            'case-control study': ['case-control', 'case control'],
            'cross-sectional study': ['cross-sectional', 'cross sectional'],
            'case series': ['case series'],
            'case report': ['case report']
        }
        
        self.UMBRAL_RELEVANCIA = 0.1
    
    def detect_section(self, text_block: str) -> str:
        text_lower = text_block.lower()[:500]
        
        for section, keywords in self.section_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return section
        
        return 'unknown'
    
    def extract_key_terms(self, hypothesis: str) -> List[str]:
        words = re.findall(r'\b[a-zA-Záéíóúñü]+\b', hypothesis.lower())
        
        filtered_words = []
        for word in words:
            if len(word) > 3 and word not in self.stop_words_es and word not in self.stop_words_en:
                if self.stemmer:
                    word = self.stemmer.stem(word)
                filtered_words.append(word)
        
        bigrams = re.findall(r'\b[a-zA-Záéíóúñü]+\s+[a-zA-Záéíóúñü]+\b', hypothesis.lower())
        for bigram in bigrams:
            if all(len(w) > 3 for w in bigram.split()):
                filtered_words.append(bigram.replace(' ', '_'))
        
        return list(set(filtered_words))
    
    def analyze_sentence_deep(self, sentence: str) -> Dict:
        sentence_lower = sentence.lower()
        
        has_negation = False
        for pattern in self.negation_patterns:
            if re.search(pattern, sentence_lower, re.IGNORECASE):
                has_negation = True
                break
        
        certainty = 'unknown'
        certainty_weight = 0.5
        for level, data in self.certainty_levels.items():
            if any(term in sentence_lower for term in data['terms']):
                certainty = level
                certainty_weight = data['weight']
                break
        
        relation = 'unknown'
        relation_weight = 0.5
        for rel_type, data in self.relation_types.items():
            if any(term in sentence_lower for term in data['terms']):
                relation = rel_type
                relation_weight = data['weight']
                break
        
        strength = relation_weight * certainty_weight
        if has_negation:
            strength *= -1
        
        return {
            'has_negation': has_negation,
            'certainty': certainty,
            'certainty_weight': certainty_weight,
            'relation': relation,
            'relation_weight': relation_weight,
            'strength': strength,
            'direction': -1 if has_negation else 1
        }
    
    def assess_study_quality(self, text: str) -> Dict:
        text_lower = text.lower()
        
        study_type = 'unknown'
        study_weight = 1.0
        for stype, patterns in self.study_type_patterns.items():
            if any(p in text_lower for p in patterns):
                study_type = stype
                study_weight = self.study_type_weights.get(stype, 1.0)
                break
        
        sample_size = None
        sample_match = re.search(r'n\s*[=:]\s*(\d+)', text_lower)
        if sample_match:
            sample_size = int(sample_match.group(1))
            if sample_size > 1000:
                study_weight *= 1.2
            elif sample_size > 500:
                study_weight *= 1.1
            elif sample_size < 50:
                study_weight *= 0.7
            elif sample_size < 100:
                study_weight *= 0.8
        
        quality_factors = []
        if re.search(r'(multicenter|multi-center)', text_lower):
            study_weight *= 1.1
            quality_factors.append('multicenter')
        if re.search(r'(double-blind|double blind|randomized)', text_lower):
            study_weight *= 1.2
            quality_factors.append('rigorous')
        if re.search(r'(prospective)', text_lower):
            study_weight *= 1.1
            quality_factors.append('prospective')
        
        return {
            'study_type': study_type,
            'type_weight': study_weight,
            'quality_factors': quality_factors,
            'sample_size': sample_size
        }
    
    def calculate_relevance(self, sentence: str, hypothesis_terms: List[str]) -> float:
        if not hypothesis_terms:
            return 0.0
        
        sentence_lower = sentence.lower()
        
        matches = sum(1 for term in hypothesis_terms if term in sentence_lower)
        
        if self.stemmer:
            sentence_stems = [self.stemmer.stem(w) for w in sentence_lower.split()]
            hypothesis_stems = [self.stemmer.stem(term.replace('_', ' ')) for term in hypothesis_terms]
            stem_matches = sum(1 for stem in hypothesis_stems if any(stem in s for s in sentence_stems))
            matches = max(matches, stem_matches)
        
        score = matches / len(hypothesis_terms)
        
        if matches >= 2:
            positions = []
            for term in hypothesis_terms:
                pos = sentence_lower.find(term)
                if pos != -1:
                    positions.append(pos)
            
            if len(positions) >= 2:
                positions.sort()
                if positions[-1] - positions[0] < 100:
                    score += 0.2
        
        return min(score, 1.0)
    
    def split_sentences(self, text: str) -> List[str]:
        if NLTK_READY:
            try:
                return sent_tokenize(text)
            except:
                return simple_sent_tokenize(text)
        else:
            return simple_sent_tokenize(text)
    
    def verify_article_text(self, text: str, hypothesis: str) -> Dict:
        if not text or len(text.strip()) < 100:
            return {
                'success': False,
                'error': 'Texto insuficiente para análisis',
                'verdict': None
            }
        
        hypothesis_terms = self.extract_key_terms(hypothesis)
        quality = self.assess_study_quality(text)
        sentences = self.split_sentences(text)
        
        block_size = max(1, len(text) // 10)
        blocks = [text[i:i+block_size] for i in range(0, len(text), block_size)]
        
        section_map = {}
        current_section = 'unknown'
        for i, block in enumerate(blocks):
            detected = self.detect_section(block)
            if detected != 'unknown':
                current_section = detected
            section_map[i] = current_section
        
        evidence_list = []
        section_counts = Counter()
        
        for i, sentence in enumerate(sentences):
            block_idx = min(i // max(1, len(sentences) // len(blocks)), len(blocks)-1)
            section = section_map.get(block_idx, 'unknown')
            section_weight = self.section_weights.get(section, 1.0)
            
            relevance = self.calculate_relevance(sentence, hypothesis_terms)
            
            if relevance > self.UMBRAL_RELEVANCIA:
                sentence_en = self.translator.translate_to_english(sentence)
                analysis = self.analyze_sentence_deep(sentence_en)
                
                if analysis['direction'] != 0:
                    evidence = {
                        'sentence': sentence[:200] + '...' if len(sentence) > 200 else sentence,
                        'sentence_en': sentence_en[:200] + '...' if len(sentence_en) > 200 else sentence_en,
                        'section': section,
                        'section_weight': section_weight,
                        'relevance': relevance,
                        'relation': analysis['relation'],
                        'certainty': analysis['certainty'],
                        'direction': analysis['direction'],
                        'strength': abs(analysis['strength']),
                        'has_negation': analysis['has_negation']
                    }
                    
                    evidence_list.append(evidence)
                    section_counts[section] += 1
        
        verdict = self.weighted_vote(evidence_list, quality)
        
        return {
            'success': True,
            'total_sentences': len(sentences),
            'relevant_sentences': len(evidence_list),
            'section_distribution': dict(section_counts),
            'study_quality': quality,
            'evidence': evidence_list[:15],
            'verdict': verdict,
            'hypothesis_terms': hypothesis_terms
        }
    
    def weighted_vote(self, evidence_list: List[Dict], quality: Dict) -> Dict:
        if not evidence_list:
            return {
                'score': 0,
                'confidence': 0,
                'verdict': 'inconclusive',
                'verdict_text': 'EVIDENCIA NO CONCLUYENTE',
                'support_count': 0,
                'against_count': 0
            }
        
        relation_weights = {
            'causal': 2.2,
            'risk': 2.0,
            'association': 1.8,
            'protective': 1.5,
            'unknown': 1.0
        }
        
        certainty_weights = {
            'definitive': 1.5,
            'strong': 1.3,
            'moderate': 1.0,
            'weak': 0.5,
            'unknown': 0.5
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        support_count = 0
        against_count = 0
        strong_evidence = []
        
        for evidence in evidence_list:
            base_weight = relation_weights.get(evidence['relation'], 1.0)
            certainty_weight = certainty_weights.get(evidence['certainty'], 0.5)
            section_weight = evidence['section_weight']
            relevance_bonus = 1 + evidence['relevance'] * 0.5
            quality_multiplier = quality['type_weight'] if quality else 1.0
            
            final_weight = (base_weight * certainty_weight * section_weight * 
                          relevance_bonus * quality_multiplier)
            
            direction = evidence['direction']
            
            if direction > 0:
                support_count += 1
                if certainty_weight > 1.0:
                    strong_evidence.append('support')
            elif direction < 0:
                against_count += 1
                if certainty_weight > 1.0:
                    strong_evidence.append('against')
            
            weighted_score += direction * final_weight
            total_weight += final_weight
        
        if total_weight == 0:
            avg_score = 0
        else:
            avg_score = weighted_score / total_weight
        
        evidence_quality = len(evidence_list) / 20.0
        strong_ratio = len(strong_evidence) / max(1, len(evidence_list))
        norm_confidence = min(1.0, (evidence_quality + strong_ratio) / 1.5)
        
        if avg_score > 0.8:
            verdict = 'strongly_supports'
            text = 'CORROBORA FUERTEMENTE'
        elif avg_score > 0.3:
            verdict = 'supports'
            text = 'CORROBORA'
        elif avg_score > -0.3:
            verdict = 'inconclusive'
            text = 'EVIDENCIA NO CONCLUYENTE'
        elif avg_score > -0.8:
            verdict = 'contradicts'
            text = 'CONTRADICE'
        else:
            verdict = 'strongly_contradicts'
            text = 'CONTRADICE FUERTEMENTE'
        
        return {
            'score': avg_score,
            'confidence': norm_confidence,
            'verdict': verdict,
            'verdict_text': text,
            'support_count': support_count,
            'against_count': against_count,
            'strong_evidence_count': len(strong_evidence)
        }

# ============================================================================
# MOTOR DE BÚSQUEDA CIENTÍFICA
# ============================================================================

class ScientificSearchEngine:
    def __init__(self, email: str):
        self.email = email
        self.delay = 0.34
    
    def _clean_query_for_general_apis(self, query: str) -> str:
        quoted_terms = re.findall(r'"([^"]*)"', query)
        
        query = re.sub(r'"[^"]*"\[[^\]]*\]', '', query)
        query = re.sub(r'\b(AND|OR|NOT)\b', ' ', query, flags=re.IGNORECASE)
        query = re.sub(r'[\(\)\[\]]', ' ', query)
        query = re.sub(r'[,;:]', ' ', query)
        query = re.sub(r'[^\w\s]', ' ', query)
        query = ' '.join(query.split())
        
        if not query or len(query) < 3:
            if quoted_terms:
                query = ' '.join(quoted_terms)
            else:
                query = "research article"
        
        return query[:500]
    
    def _extract_keywords_for_crossref(self, query: str) -> str:
        return self._clean_query_for_general_apis(query)
    
    def _extract_keywords_for_openalex(self, query: str) -> str:
        return self._clean_query_for_general_apis(query)
    
    def _extract_keywords_for_europepmc(self, query: str, year_range: tuple = None) -> str:
        clean_query = self._clean_query_for_general_apis(query)
        
        if year_range and year_range[0] and year_range[1]:
            date_filter = f" AND (PUB_YEAR:{year_range[0]}-{year_range[1]})"
            return f"({clean_query}){date_filter}"
        
        return clean_query
    
    def format_pubmed_query(self, query: str, year_range: tuple = None) -> str:
        query = ' '.join(query.split())
        
        mesh_pattern = r'"([^"]+)"\[Mesh\]'
        
        def replace_mesh(match):
            term = match.group(1).strip()
            return f'"{term}"[mh]'
        
        query = re.sub(mesh_pattern, replace_mesh, query)
        
        field_mappings = {
            '[Publication Type]': '[pt]',
            '[Title/Abstract]': '[tiab]',
            '[Author]': '[au]',
            '[Journal]': '[ta]',
            '[Date - Publication]': '[dp]',
            '[MeSH Major Topic]': '[majr]',
            '[MeSH Subheading]': '[sh]'
        }
        
        for old, new in field_mappings.items():
            query = query.replace(old, new)
        
        boolean_ops = ['and', 'or', 'not']
        for op in boolean_ops:
            query = re.sub(rf'\b{op}\b', op.upper(), query, flags=re.IGNORECASE)
        
        if year_range and year_range[0] and year_range[1]:
            year_filter = f" AND ({year_range[0]}[pdat] : {year_range[1]}[pdat])"
            query = f"({query}){year_filter}"
        
        return query
    
    def search_pubmed_advanced(self, query: str, max_results: int = 1000, year_range: tuple = None) -> list:
        all_results = []
        
        try:
            formatted_query = self.format_pubmed_query(query, year_range)
            
            if st.session_state.get('debug_mode', False):
                st.write(f"🔍 PubMed query: {formatted_query}")
            
            encoded_query = urllib.parse.quote(formatted_query, safe='()" ')
            
            search_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                f"?db=pubmed"
                f"&term={encoded_query}"
                f"&retmode=json"
                f"&retmax=0"
                f"&usehistory=y"
                f"&tool=streamlit_app"
                f"&email={self.email}"
            )
            
            time.sleep(self.delay)
            response = requests.get(search_url, timeout=30)
            response.raise_for_status()
            search_data = response.json()
            
            if 'esearchresult' not in search_data:
                return []
            
            webenv = search_data.get('esearchresult', {}).get('webenv')
            query_key = search_data.get('esearchresult', {}).get('querykey')
            count = int(search_data.get('esearchresult', {}).get('count', 0))
            
            if st.session_state.get('debug_mode', False):
                st.write(f"📊 PubMed: {count} resultados encontrados")
            
            if not webenv or not query_key or count == 0:
                return []
            
            retmax = min(max_results, count)
            batch_size = 100
            
            for retstart in range(0, retmax, batch_size):
                current_batch_size = min(batch_size, retmax - retstart)
                
                fetch_ids_url = (
                    f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                    f"?db=pubmed"
                    f"&WebEnv={webenv}"
                    f"&query_key={query_key}"
                    f"&retmode=json"
                    f"&retstart={retstart}"
                    f"&retmax={current_batch_size}"
                    f"&tool=streamlit_app"
                    f"&email={self.email}"
                )
                
                time.sleep(self.delay)
                ids_response = requests.get(fetch_ids_url, timeout=30)
                ids_response.raise_for_status()
                ids_data = ids_response.json()
                
                id_list = ids_data.get('esearchresult', {}).get('idlist', [])
                
                if id_list:
                    batch_results = self.fetch_pubmed_batch(id_list)
                    all_results.extend(batch_results)
            
            return all_results
            
        except Exception as e:
            st.warning(f"Error en PubMed: {str(e)}")
            return []
    
    def fetch_pubmed_batch(self, ids_batch: list) -> list:
        results = []
        
        try:
            ids = ','.join(ids_batch)
            fetch_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                f"?db=pubmed"
                f"&id={ids}"
                f"&retmode=xml"
                f"&rettype=abstract"
                f"&tool=streamlit_app"
                f"&email={self.email}"
            )
            
            time.sleep(self.delay)
            response = requests.get(fetch_url, timeout=30)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            
            for article in root.findall('.//PubmedArticle'):
                try:
                    title_elem = article.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None and title_elem.text else "Título no disponible"
                    
                    authors = []
                    author_list = article.findall('.//Author')
                    for author in author_list[:5]:
                        last = author.find('LastName')
                        fore = author.find('ForeName')
                        if last is not None and last.text:
                            if fore is not None and fore.text:
                                authors.append(f"{fore.text} {last.text}")
                            else:
                                authors.append(last.text)
                    
                    journal_elem = article.find('.//Journal/Title')
                    journal = journal_elem.text if journal_elem is not None and journal_elem.text else ""
                    
                    year = ""
                    year_elem = article.find('.//PubDate/Year')
                    if year_elem is not None and year_elem.text:
                        year = year_elem.text
                    else:
                        medline_date = article.find('.//PubDate/MedlineDate')
                        if medline_date is not None and medline_date.text:
                            year_match = re.search(r'\b(19|20)\d{2}\b', medline_date.text)
                            if year_match:
                                year = year_match.group()
                    
                    doi = ""
                    doi_elem = article.find(".//ArticleId[@IdType='doi']")
                    if doi_elem is not None and doi_elem.text:
                        doi = doi_elem.text
                    
                    pmid = ""
                    pmid_elem = article.find(".//PMID")
                    if pmid_elem is not None and pmid_elem.text:
                        pmid = pmid_elem.text
                    
                    abstract = ""
                    abstract_parts = []
                    for abstract_elem in article.findall('.//Abstract/AbstractText'):
                        if abstract_elem.text:
                            label = abstract_elem.get('Label', '')
                            if label:
                                abstract_parts.append(f"{label}: {abstract_elem.text}")
                            else:
                                abstract_parts.append(abstract_elem.text)
                    
                    if abstract_parts:
                        abstract = ' '.join(abstract_parts)
                    else:
                        abstract_elem = article.find('.//AbstractText')
                        if abstract_elem is not None and abstract_elem.text:
                            abstract = abstract_elem.text
                    
                    results.append({
                        'base_datos': 'PubMed',
                        'titulo': title,
                        'autores': ', '.join(authors)[:200],
                        'revista': journal,
                        'año': year,
                        'doi': doi,
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                        'pmid': pmid,
                        'abstract': abstract[:500] + '...' if len(abstract) > 500 else abstract,
                        'tipo': 'Artículo'
                    })
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            st.warning(f"Error en fetch_pubmed_batch: {str(e)}")
        
        return results
    
    def search_crossref(self, query: str, max_results: int = 1000, year_range: tuple = None) -> list:
        results = []
        try:
            simple_query = self._extract_keywords_for_crossref(query)
            
            url = "https://api.crossref.org/works"
            params = {
                'query': simple_query,
                'rows': min(1000, max_results),
                'sort': 'relevance',
                'order': 'desc'
            }
            
            if year_range and year_range[0] and year_range[1]:
                params['filter'] = f"from-pub-date:{year_range[0]},until-pub-date:{year_range[1]}"
            
            time.sleep(self.delay)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for item in data.get('message', {}).get('items', []):
                year = ''
                if item.get('published-print'):
                    year = str(item['published-print']['date-parts'][0][0])
                elif item.get('published-online'):
                    year = str(item['published-online']['date-parts'][0][0])
                
                authors = []
                for author in item.get('author', [])[:5]:
                    if 'given' in author and 'family' in author:
                        authors.append(f"{author['given']} {author['family']}")
                    elif 'family' in author:
                        authors.append(author['family'])
                
                results.append({
                    'base_datos': 'CrossRef',
                    'titulo': item.get('title', ['Título no disponible'])[0],
                    'autores': ', '.join(authors)[:200],
                    'revista': item.get('container-title', [''])[0] if item.get('container-title') else '',
                    'año': year,
                    'doi': item.get('DOI', ''),
                    'url': f"https://doi.org/{item['DOI']}" if item.get('DOI') else '',
                    'tipo': item.get('type', '')
                })
                
        except Exception as e:
            st.warning(f"Error en CrossRef: {str(e)}")
        
        return results[:max_results]
    
    def search_openalex(self, query: str, max_results: int = 1000, year_range: tuple = None) -> list:
        results = []
        try:
            simple_query = self._extract_keywords_for_openalex(query)
            
            url = "https://api.openalex.org/works"
            params = {
                'search': simple_query,
                'per-page': 100,
                'sort': 'relevance_score:desc'
            }
            
            if year_range and year_range[0] and year_range[1]:
                params['filter'] = f"publication_year:{year_range[0]}-{year_range[1]}"
            
            page = 1
            while len(results) < max_results:
                params['page'] = page
                
                time.sleep(self.delay)
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                page_results = data.get('results', [])
                if not page_results:
                    break
                
                for item in page_results:
                    titulo = item.get('title')
                    if titulo is None:
                        titulo = "Título no disponible"
                    
                    autores_list = []
                    for a in item.get('authorships', [])[:5]:
                        author_data = a.get('author', {})
                        if author_data is not None:
                            author_name = author_data.get('display_name')
                            if author_name:
                                autores_list.append(author_name)
                    
                    revista = ""
                    host_venue = item.get('host_venue')
                    if host_venue is not None:
                        revista = host_venue.get('display_name', '')
                    
                    año = item.get('publication_year')
                    año_str = str(año) if año is not None else ""
                    
                    doi = item.get('doi')
                    if doi is not None:
                        doi = doi.replace('https://doi.org/', '')
                    else:
                        doi = ""
                    
                    results.append({
                        'base_datos': 'OpenAlex',
                        'titulo': titulo,
                        'autores': ', '.join(autores_list),
                        'revista': revista,
                        'año': año_str,
                        'doi': doi,
                        'url': f"https://doi.org/{doi}" if doi else "",
                        'tipo': item.get('type', ''),
                    })
                    
                    if len(results) >= max_results:
                        break
                
                page += 1
                if page > 10:
                    break
                
        except Exception as e:
            st.warning(f"Error en OpenAlex: {str(e)}")
        
        return results[:max_results]
    
    def search_europe_pmc(self, query: str, max_results: int = 1000, year_range: tuple = None) -> list:
        results = []
        try:
            europe_query = self._extract_keywords_for_europepmc(query, year_range)
            
            url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            
            page_size = 100
            page = 1
            
            while len(results) < max_results:
                params = {
                    'query': europe_query,
                    'format': 'json',
                    'pageSize': page_size,
                    'page': page,
                    'resultType': 'core'
                }
                
                time.sleep(self.delay)
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                page_results = data.get('resultList', {}).get('result', [])
                if not page_results:
                    break
                
                for item in page_results:
                    results.append({
                        'base_datos': 'Europe PMC',
                        'titulo': item.get('title', 'Título no disponible'),
                        'autores': ', '.join([a.get('fullName', '') for a in item.get('authorList', {}).get('author', [])[:5]]),
                        'revista': item.get('journalTitle', ''),
                        'año': str(item.get('pubYear', '')),
                        'doi': item.get('doi', ''),
                        'url': f"https://europepmc.org/article/{item.get('source', '')}/{item.get('id', '')}",
                        'pmid': item.get('pmid', ''),
                        'pmcid': item.get('pmcid', ''),
                        'abstract': item.get('abstractText', '')[:300] + '...' if item.get('abstractText') else '',
                        'tipo': 'Artículo'
                    })
                    
                    if len(results) >= max_results:
                        break
                
                page += 1
                if page > 10:
                    break
                
        except Exception as e:
            st.warning(f"Error en Europe PMC: {str(e)}")
        
        return results[:max_results]
    
    def search_all(self, query: str, max_results_per_db: int = 1000, selected_dbs: list = None, 
                   year_range: tuple = None) -> pd.DataFrame:
        if selected_dbs is None:
            selected_dbs = ['PubMed', 'CrossRef', 'OpenAlex', 'Europe PMC']
        
        all_results = []
        
        search_functions = {
            'PubMed': self.search_pubmed_advanced,
            'CrossRef': self.search_crossref,
            'OpenAlex': self.search_openalex,
            'Europe PMC': self.search_europe_pmc
        }
        
        for db in selected_dbs:
            if db in search_functions:
                try:
                    with st.spinner(f"🔍 Buscando en {db}..."):
                        results = search_functions[db](query, max_results_per_db, year_range)
                        all_results.extend(results)
                        st.success(f"✅ {db}: {len(results)} resultados")
                except Exception as e:
                    st.warning(f"Error en {db}: {str(e)}")
        
        if all_results:
            df = pd.DataFrame(all_results)
            
            if 'año' in df.columns:
                df['año'] = pd.to_numeric(df['año'], errors='coerce')
            
            if 'doi' in df.columns:
                df = df.drop_duplicates(subset=['doi'], keep='first')
            
            return df
        else:
            return pd.DataFrame()

# ============================================================================
# OBTENEDOR DE TEXTO DE ARTÍCULOS
# ============================================================================

class ArticleTextFetcher:
    def __init__(self):
        self.session = requests.Session()
        retry_strategy = Retry(
            total=2,
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
        if not doi or pd.isna(doi) or doi == '':
            return None, "DOI vacío"
        
        doi = str(doi).strip()
        if doi.startswith('https://doi.org/'):
            doi = doi.replace('https://doi.org/', '')
        elif doi.startswith('doi:'):
            doi = doi.replace('doi:', '')
        
        self.wait()
        
        try:
            url = f"https://api.openalex.org/works/doi:{doi}"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                oa_url = data.get('open_access', {}).get('oa_url')
                if oa_url:
                    return f"Open Access disponible en: {oa_url}", f"OpenAlex (OA URL)"
        except:
            pass
        
        try:
            url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:{doi}&format=json"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = data.get('resultList', {}).get('result', [])
                if results:
                    abstract = results[0].get('abstractText', '')
                    if abstract:
                        return abstract, "Europe PMC (Abstract)"
        except:
            pass
        
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

# ============================================================================
# CLASE PRINCIPAL INTEGRADA
# ============================================================================

class IntegratedScientificVerifier:
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
        self.results = []
        self.stats = {k: 0 for k in self.stats}
        
        if progress_callback:
            progress_callback("🔍 Buscando artículos en bases de datos...", 0.05)
        
        articles_df = self.search_engine.search_all(
            query, max_results_per_db, selected_dbs, year_range
        )
        
        if articles_df.empty:
            return pd.DataFrame()
        
        self.stats['total_articles'] = len(articles_df)
        
        if progress_callback:
            progress_callback(f"✅ Encontrados {len(articles_df)} artículos. Iniciando análisis...", 0.1)
        
        results_list = []
        total_articles = len(articles_df)
        
        for idx, row in articles_df.iterrows():
            current = idx + 1
            if progress_callback:
                progress_value = 0.1 + 0.85 * (current / total_articles)
                progress_value = min(0.95, max(0.1, progress_value))
                progress_callback(
                    f"🔬 Analizando artículo {current}/{total_articles}: {str(row['titulo'])[:50]}...", 
                    progress_value
                )
            
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
                'oraciones_totales': 0,
                'oraciones_relevantes': 0,
                'detalle_evidencia': ''
            }
            
            if article_text:
                self.stats['with_text'] += 1
                
                analysis = self.semantic_verifier.verify_article_text(article_text, hypothesis)
                
                if analysis['success']:
                    verdict = analysis['verdict']
                    
                    result_row.update({
                        'veredicto': verdict['verdict_text'],
                        'confianza': verdict['confidence'],
                        'puntuacion': verdict['score'],
                        'evidencia_a_favor': verdict['support_count'],
                        'evidencia_en_contra': verdict['against_count'],
                        'oraciones_totales': analysis['total_sentences'],
                        'oraciones_relevantes': analysis['relevant_sentences']
                    })
                    
                    self.stats['analyzed'] += 1
                    if 'CORROBORA' in verdict['verdict_text']:
                        self.stats['corroboran'] += 1
                    elif 'CONTRADICE' in verdict['verdict_text']:
                        self.stats['contradicen'] += 1
                    else:
                        self.stats['inconclusos'] += 1
                    
                    if analysis['evidence']:
                        ev_summary = []
                        for ev in analysis['evidence'][:3]:
                            ev_summary.append(f"[{ev['section']}] {ev['relation']} ({ev['certainty']})")
                        result_row['detalle_evidencia'] = ' | '.join(ev_summary)
            else:
                result_row['veredicto'] = 'TEXTO NO DISPONIBLE'
            
            results_list.append(result_row)
            time.sleep(0.1)
        
        if progress_callback:
            progress_callback("✅ Análisis completado", 1.0)
        
        self.results = pd.DataFrame(results_list)
        return self.results
    
    def generate_report(self) -> str:
        if self.results.empty:
            return "No hay resultados para generar reporte."
        
        report = []
        report.append("="*80)
        report.append("REPORTE DE VERIFICACIÓN SEMÁNTICA INTEGRADA")
        report.append("="*80)
        report.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total artículos encontrados: {self.stats['total_articles']}")
        report.append(f"Artículos con texto disponible: {self.stats['with_text']}")
        report.append(f"Artículos analizados: {self.stats['analyzed']}")
        report.append("")
        report.append("RESULTADOS GLOBALES:")
        report.append(f"✅ Corroboran: {self.stats['corroboran']}")
        report.append(f"❌ Contradicen: {self.stats['contradicen']}")
        report.append(f"⚠️ Inconclusos: {self.stats['inconclusos']}")
        report.append("")
        report.append("DETALLE POR ARTÍCULO:")
        report.append("-"*80)
        
        for idx, row in self.results.iterrows():
            report.append(f"\n📄 {row['titulo']}")
            report.append(f"   Base: {row['base_datos']} | Año: {row['año']}")
            report.append(f"   DOI: {row['doi']}")
            
            if row['veredicto'] == 'TEXTO NO DISPONIBLE':
                report.append(f"   ⚠️ {row['veredicto']} - {row['fuente_texto']}")
            else:
                report.append(f"   Veredicto: {row['veredicto']} (Confianza: {row['confianza']:.1%})")
                report.append(f"   Evidencia: {row['evidencia_a_favor']} a favor, {row['evidencia_en_contra']} en contra")
                if row['detalle_evidencia']:
                    report.append(f"   Evidencia destacada: {row['detalle_evidencia']}")
        
        return "\n".join(report)
    
    def generate_html_report(self) -> str:
        if self.results.empty:
            return "<p>No hay resultados para generar reporte.</p>"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #1E88E5; }}
                h2 {{ color: #333; border-bottom: 2px solid #1E88E5; padding-bottom: 5px; }}
                .stats {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .verdict-assert {{ color: #4CAF50; font-weight: bold; }}
                .verdict-reject {{ color: #f44336; font-weight: bold; }}
                .verdict-inconclusive {{ color: #ff9800; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #1E88E5; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>🔬 Reporte de Verificación Semántica</h1>
            <p><strong>Fecha:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>📊 Resultados Globales</h2>
            <div class="stats">
                <p><strong>Total artículos encontrados:</strong> {self.stats['total_articles']}</p>
                <p><strong>Artículos con texto disponible:</strong> {self.stats['with_text']}</p>
                <p><strong>Artículos analizados:</strong> {self.stats['analyzed']}</p>
                <p><strong>✅ Corroboran:</strong> {self.stats['corroboran']}</p>
                <p><strong>❌ Contradicen:</strong> {self.stats['contradicen']}</p>
                <p><strong>⚠️ Inconclusos:</strong> {self.stats['inconclusos']}</p>
            </div>
            
            <h2>📋 Detalle de Artículos</h2>
            <table>
                <tr>
                    <th>Base</th>
                    <th>Título</th>
                    <th>Año</th>
                    <th>Veredicto</th>
                    <th>Confianza</th>
                </tr>
        """
        
        for idx, row in self.results.iterrows():
            titulo = str(row.get('titulo', '')) if pd.notna(row.get('titulo')) else "Título no disponible"
            base_datos = str(row.get('base_datos', '')) if pd.notna(row.get('base_datos')) else "Desconocida"
            año = str(row.get('año', '')) if pd.notna(row.get('año')) else ""
            veredicto = str(row.get('veredicto', '')) if pd.notna(row.get('veredicto')) else "NO DISPONIBLE"
            confianza = float(row.get('confianza', 0)) if pd.notna(row.get('confianza')) else 0
            
            verdict_class = ""
            if 'CORROBORA' in veredicto:
                verdict_class = "verdict-assert"
            elif 'CONTRADICE' in veredicto:
                verdict_class = "verdict-reject"
            elif 'NO CONCLUYENTE' in veredicto:
                verdict_class = "verdict-inconclusive"
            
            html += f"""
                <tr>
                    <td>{base_datos}</td>
                    <td>{titulo[:100]}...</td>
                    <td>{año}</td>
                    <td class="{verdict_class}">{veredicto}</td>
                    <td>{confianza:.1%}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html

# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def get_badge_class(db_name: str) -> str:
    classes = {
        'PubMed': 'badge-pubmed',
        'CrossRef': 'badge-crossref',
        'OpenAlex': 'badge-openalex',
        'Europe PMC': 'badge-europepmc'
    }
    return classes.get(db_name, 'badge-pubmed')

def get_verdict_class(verdict: str) -> str:
    if 'CORROBORA' in verdict:
        return 'verdict-assert'
    elif 'CONTRADICE' in verdict:
        return 'verdict-reject'
    elif 'NO CONCLUYENTE' in verdict:
        return 'verdict-inconclusive'
    else:
        return ''

# ============================================================================
# FUNCIÓN PARA ENVIAR RESULTADOS POR CORREO
# ============================================================================

def enviar_resultados_email(destinatario, integrator):
    if integrator is None or integrator.results is None or integrator.results.empty:
        st.warning("No hay resultados para enviar por correo.")
        return False
    
    reporte_txt = integrator.generate_report()
    reporte_html = integrator.generate_html_report()
    
    archivos = []
    
    csv_buffer = io.BytesIO()
    integrator.results.to_csv(csv_buffer, index=False, encoding='utf-8')
    archivos.append({
        'nombre': f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        'contenido': csv_buffer.getvalue()
    })
    
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        integrator.results.to_excel(writer, sheet_name='Resultados', index=False)
        summary = pd.DataFrame([integrator.stats])
        summary.to_excel(writer, sheet_name='Resumen', index=False)
    archivos.append({
        'nombre': f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        'contenido': excel_buffer.getvalue()
    })
    
    asunto = f"🔬 Reporte de Verificación Semántica - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    mensaje_html = f"""
    <html>
    <body>
        <h2>🔬 Reporte de Verificación Semántica</h2>
        <p>Estimado usuario,</p>
        <p>Adjunto encontrarás los resultados completos de tu análisis de verificación semántica.</p>
        
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
            Este es un mensaje automático del Buscador y Verificador Semántico Integrado.
        </p>
    </body>
    </html>
    """
    
    mensaje_texto = f"""
    Reporte de Verificación Semántica
    
    Resumen:
    - Total artículos: {integrator.stats['total_articles']}
    - Artículos analizados: {integrator.stats['analyzed']}
    - ✅ Corroboran: {integrator.stats['corroboran']}
    - ❌ Contradicen: {integrator.stats['contradicen']}
    - ⚠️ Inconclusos: {integrator.stats['inconclusos']}
    
    Los archivos adjuntos contienen el detalle completo.
    """
    
    return enviar_correo(
        destinatario=destinatario,
        asunto=asunto,
        mensaje_html=mensaje_html,
        mensaje_texto=mensaje_texto,
        archivos=archivos
    )

# ============================================================================
# INTERFAZ PRINCIPAL DE STREAMLIT (VERSIÓN CORREGIDA CON PERSISTENCIA)
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">🔬 Buscador y Verificador Semántico Integrado</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ALTO VOLUMEN: Hasta 1000 artículos por base • Análisis AI automático • VERSIÓN COMPLETA CORREGIDA</p>', 
                unsafe_allow_html=True)
    
    # Inicializar session state con TODAS las variables necesarias
    if 'query' not in st.session_state:
        st.session_state['query'] = ""
    if 'hypothesis' not in st.session_state:
        st.session_state['hypothesis'] = ""
    if 'hypothesis_en' not in st.session_state:
        st.session_state['hypothesis_en'] = ""
    if 'user_email' not in st.session_state:
        st.session_state['user_email'] = ""
    if 'integrator' not in st.session_state:
        st.session_state['integrator'] = None
    if 'debug_mode' not in st.session_state:
        st.session_state['debug_mode'] = False
    
    # NUEVAS VARIABLES PARA PERSISTENCIA
    if 'analysis_completed' not in st.session_state:
        st.session_state['analysis_completed'] = False
    if 'last_results_df' not in st.session_state:
        st.session_state['last_results_df'] = None
    if 'last_stats' not in st.session_state:
        st.session_state['last_stats'] = None
    if 'elapsed_time' not in st.session_state:
        st.session_state['elapsed_time'] = 0
    
    hypothesis_assistant = HypothesisAssistant()
    hypothesis_assistant.render_assistant_ui()
    
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
        st.markdown("## ⚙️ Configuración")
        
        email = st.text_input(
            "📧 Email (requerido para NCBI y para recibir resultados)", 
            value=st.session_state.get('user_email', ''),
            placeholder="tu@email.com",
            key="email_input"
        )
        st.session_state['user_email'] = email
        
        debug_mode = st.checkbox(
            "🔧 Modo depuración",
            value=st.session_state.get('debug_mode', False),
            key="debug_checkbox"
        )
        st.session_state['debug_mode'] = debug_mode
        
        if NLTK_READY:
            st.success("✅ NLTK configurado correctamente")
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
            min_value=10, max_value=1000, value=100, step=10
        )
        
        st.markdown("### 🔧 Umbrales de análisis")
        min_relevance = st.slider(
            "Relevancia mínima",
            min_value=0.05,
            max_value=0.3,
            value=0.1,
            step=0.05
        )
        
        st.markdown("### ⚠️ Advertencia")
        st.warning("""
        **Tiempo de procesamiento:**
        - 50 artículos: ~2-3 minutos
        - 100 artículos: ~5-7 minutos
        - 200 artículos: ~10-15 minutos
        - 500 artículos: ~25-35 minutos
        - 1000 artículos: ~50-70 minutos
        """)
        
        st.markdown("### 📋 Ejemplos")
        
        # EJEMPLOS OPTIMIZADOS
        if st.button("Cargar ejemplo: Ticagrelor y disnea"):
            st.session_state['query'] = '("Ticagrelor"[Mesh] AND "Dyspnea"[Mesh] AND "Myocardial Ischemia"[Mesh])'
            st.session_state['hypothesis'] = "El ticagrelor causa disnea como efecto secundario en pacientes con cardiopatía isquémica"
            st.rerun()
        
        if st.button("Cargar ejemplo: Ruptura cardiaca postinfarto"):
            # CONSULTA OPTIMIZADA CON TUS KEYWORDS
            st.session_state['query'] = '("Heart Rupture, Post-Infarction"[Mesh] OR "heart rupture"[Title/Abstract] OR "cardiac rupture"[Title/Abstract]) AND ("Myocardial Infarction"[Mesh]) AND ("intramyocardial dissection"[Title/Abstract] OR "intramyocardial hematoma"[Title/Abstract] OR "myocardial complex rupture"[Title/Abstract] OR "anatomical patterns"[Title/Abstract] OR "myocardial structure"[Title/Abstract] OR "ventricular septal rupture"[MeSH] OR "free wall rupture"[Title/Abstract])'
            st.session_state['hypothesis'] = "En la ruptura cardiaca postinfarto, el corazón se rompe siguiendo patrones anatómicos reconocibles (disección intramiocárdica, hematoma intramiocárdico o ruptura compleja)"
            st.rerun()
        
        if st.button("Cargar ejemplo: Ejercicio y diabetes"):
            st.session_state['query'] = '("Exercise"[Mesh] AND "Diabetes Mellitus, Type 2"[Mesh] AND "prevention and control"[Subheading])'
            st.session_state['hypothesis'] = "El ejercicio físico regular reduce la incidencia de diabetes tipo 2 en adultos con sobrepeso"
            st.rerun()
    
    # Área principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_query = st.text_area(
            "🔍 Consulta de búsqueda:",
            value=st.session_state['query'],
            height=120,
            placeholder='Ej: ("Ticagrelor"[Mesh] AND "Myocardial Ischemia"[Mesh] AND "Dyspnea"[Mesh])',
            key="query_input"
        )
        if search_query != st.session_state['query']:
            st.session_state['query'] = search_query
            # Si cambia la consulta, reiniciar estado de análisis
            st.session_state['analysis_completed'] = False
    
    with col2:
        hypothesis_es = st.text_area(
            "🔬 Conjetura a verificar (español):",
            value=st.session_state['hypothesis'],
            height=68,
            placeholder='Ej: El fármaco X causa efecto Y en pacientes con Z',
            key="hypothesis_input"
        )
        
        if hypothesis_es != st.session_state.get('hypothesis', ''):
            st.session_state['hypothesis'] = hypothesis_es
            translator = TranslationManager()
            st.session_state['hypothesis_en'] = translator.translate_to_english(hypothesis_es)
            # Si cambia la hipótesis, reiniciar estado de análisis
            st.session_state['analysis_completed'] = False
        
        if st.session_state.get('hypothesis_en'):
            st.markdown(f"**🇬🇧 Inglés (para búsqueda):**")
            st.info(st.session_state['hypothesis_en'][:150] + ('...' if len(st.session_state['hypothesis_en']) > 150 else ''))
    
    col1, col2, col3 = st.columns(3)
    with col2:
        analyze_button = st.button("🚀 INICIAR ANÁLISIS INTEGRADO", type="primary", use_container_width=True)
    
    # ============================================================================
    # BLOQUE DE ANÁLISIS (SOLO SE EJECUTA CUANDO SE HACE CLIC EN EL BOTÓN)
    # ============================================================================
    
    if analyze_button and search_query and hypothesis_es:
        user_email = st.session_state.get('user_email', '')
        if not user_email or not validate_email(user_email):
            st.error("❌ Ingresa un email válido en la barra lateral")
        else:
            selected_dbs = [db for db, selected in databases.items() if selected]
            
            if not selected_dbs:
                st.error("❌ Selecciona al menos una base de datos")
            else:
                integrator = IntegratedScientificVerifier(user_email)
                integrator.semantic_verifier.UMBRAL_RELEVANCIA = min_relevance
                
                hypothesis_for_analysis = st.session_state.get('hypothesis_en', hypothesis_es)
                
                progress_container = st.container()
                with progress_container:
                    st.markdown('<div class="progress-container">', unsafe_allow_html=True)
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    time_estimate = st.empty()
                    st.markdown('</div>', unsafe_allow_html=True)
                
                start_time = time.time()
                
                def update_progress(message, value):
                    status_text.text(message)
                    value = max(0.0, min(1.0, value))
                    progress_bar.progress(value)
                    elapsed = time.time() - start_time
                    if value > 0:
                        estimated_total = elapsed / value
                        remaining = estimated_total * (1 - value)
                        if remaining > 0:
                            time_estimate.text(f"⏱️ Tiempo transcurrido: {elapsed:.1f}s | Estimado restante: {remaining:.1f}s")
                
                with st.spinner("Ejecutando análisis integrado..."):
                    results_df = integrator.run_analysis(
                        search_query, 
                        hypothesis_for_analysis, 
                        max_results, 
                        selected_dbs, 
                        year_range,
                        update_progress
                    )
                    elapsed_time = time.time() - start_time
                
                # GUARDAR TODO EN SESSION STATE PARA PERSISTENCIA
                if not results_df.empty:
                    st.session_state['integrator'] = integrator
                    st.session_state['last_results_df'] = results_df.copy()
                    st.session_state['last_stats'] = integrator.stats.copy()
                    st.session_state['analysis_completed'] = True
                    st.session_state['elapsed_time'] = elapsed_time
                    
                    # Forzar rerun para mostrar resultados
                    st.rerun()
                else:
                    st.warning("😕 No se encontraron artículos. Prueba con otra consulta o amplía el rango de años.")
    
    # ============================================================================
    # BLOQUE DE VISUALIZACIÓN DE RESULTADOS (SIEMPRE SE MUESTRA SI HAY RESULTADOS)
    # ============================================================================
    
    if st.session_state.get('analysis_completed', False) and st.session_state.get('last_results_df') is not None:
        integrator = st.session_state['integrator']
        results_df = st.session_state['last_results_df']
        stats = st.session_state['last_stats']
        elapsed_time = st.session_state['elapsed_time']
        
        if not results_df.empty:
            st.success(f"✅ Análisis completado en {elapsed_time:.1f} segundos ({elapsed_time/60:.1f} minutos)")
            
            st.markdown("## 📊 RESULTADOS GLOBALES")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Artículos encontrados", stats['total_articles'])
            with col2:
                st.metric("Con texto", stats['with_text'])
            with col3:
                st.metric("✅ Corroboran", stats['corroboran'])
            with col4:
                st.metric("❌ Contradicen", stats['contradicen'])
            with col5:
                st.metric("⚠️ Inconclusos", stats['inconclusos'])
            
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
            
            st.markdown("## 📋 ARTÍCULOS ANALIZADOS")
            
            display_cols = ['base_datos', 'titulo', 'año', 'veredicto', 'confianza', 
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
                    'veredicto': 'Veredicto',
                    'confianza': 'Confianza',
                    'evidencia_a_favor': 'A favor',
                    'evidencia_en_contra': 'En contra'
                }
            )
            
            st.markdown("## 🔍 DETALLE DE ANÁLISIS (primeros 10 artículos)")
            
            for idx, row in results_df.head(10).iterrows():
                badge_class = get_badge_class(row['base_datos'])
                
                if row['veredicto'] == 'TEXTO NO DISPONIBLE':
                    st.markdown(f"""
                    <div class="result-card">
                        <span class="{badge_class}">{row['base_datos']}</span>
                        <div class="result-title">{row['titulo']}</div>
                        <div class="result-meta">
                            <b>Año:</b> {row.get('año', 'No disponible')}<br>
                            <b>DOI:</b> {row.get('doi', 'No disponible')}<br>
                            <b>Estado:</b> ⚠️ TEXTO NO DISPONIBLE - {row['fuente_texto']}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    verdict_class = get_verdict_class(row['veredicto'])
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <span class="{badge_class}">{row['base_datos']}</span>
                        <div class="result-title">{row['titulo']}</div>
                        <div class="result-meta">
                            <b>Año:</b> {row.get('año', 'No disponible')}<br>
                            <b>DOI:</b> {row.get('doi', 'No disponible')}<br>
                            <span class="{verdict_class}">{row['veredicto']}</span> (Confianza: {row['confianza']:.1%})<br>
                            <b>Evidencia:</b> {row['evidencia_a_favor']} a favor, {row['evidencia_en_contra']} en contra
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if row['detalle_evidencia']:
                        st.markdown(f"""
                        <div class="evidence-box">
                            <b>Evidencia destacada:</b> {row['detalle_evidencia']}
                        </div>
                        """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 5])
                with col1:
                    if row.get('url') and pd.notna(row['url']):
                        st.link_button("🔗 Ver", row['url'])
                with col2:
                    if row.get('doi') and pd.notna(row['doi']):
                        doi_link = f"https://doi.org/{row['doi']}"
                        st.link_button("📋 DOI", doi_link)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            if len(results_df) > 10:
                st.info(f"... y {len(results_df) - 10} artículos más. Exporta los resultados para ver el listado completo.")
            
            st.markdown("## 💾 EXPORTAR RESULTADOS")
            
            col1, col2, col3 = st.columns(3)
            
            report_text = integrator.generate_report() if integrator else "No disponible"
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
                    
                    summary = pd.DataFrame([stats])
                    summary.to_excel(writer, sheet_name='Resumen', index=False)
                    
                    if not results_df[results_df['veredicto'] != 'TEXTO NO DISPONIBLE'].empty:
                        verdict_stats = results_df[results_df['veredicto'] != 'TEXTO NO DISPONIBLE']['veredicto'].value_counts().reset_index()
                        verdict_stats.columns = ['Veredicto', 'Cantidad']
                        verdict_stats.to_excel(writer, sheet_name='Por Veredicto', index=False)
                
                st.download_button(
                    "📥 Excel",
                    buffer.getvalue(),
                    file_name=f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
    
    # Sección para enviar resultados por correo
    if st.session_state.get('analysis_completed', False) and st.session_state.get('integrator') is not None:
        st.markdown("---")
        st.markdown("## 📧 ENVIAR RESULTADOS POR CORREO")
        st.markdown('<div class="email-box">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📨 ENVIAR RESULTADOS A MI CORREO", type="primary", use_container_width=True):
                with st.spinner("Enviando resultados por correo..."):
                    if enviar_resultados_email(st.session_state['user_email'], st.session_state['integrator']):
                        st.success(f"✅ Resultados enviados correctamente a {st.session_state['user_email']}")
                        st.balloons()
                    else:
                        st.error("❌ No se pudo enviar el correo. Verifica la configuración.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🔬 Buscador y Verificador Semántico Integrado v3.6 | CORREGIDO: Persistencia de resultados • PubMed con comas • Ejemplos optimizados</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
