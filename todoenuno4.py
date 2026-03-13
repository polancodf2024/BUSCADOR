"""
Buscador y Verificador Semántico Integrado - VERSIÓN ALTO VOLUMEN CON IA AVANZADA
Combina búsqueda en múltiples bases de datos científicas con análisis semántico AI de última generación
Soporta hasta 1000 resultados por base de datos
"""

import streamlit as st
import pandas as pd
import requests
import time
import urllib.parse
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import io
import xml.etree.ElementTree as ET
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import os
import pathlib
from typing import List, Dict, Tuple, Optional
import zipfile
import random
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        # Configuración SMTP (usando variables de entorno o secrets de Streamlit)
        self.SMTP_SERVER = st.secrets.get("smtp_server", "smtp.gmail.com")
        self.SMTP_PORT = st.secrets.get("smtp_port", 587)
        self.EMAIL_USER = st.secrets.get("email_user", "")
        self.EMAIL_PASSWORD = st.secrets.get("email_password", "")
        self.MAX_FILE_SIZE_MB = 10
        
        # Verificar si tenemos configuración de correo
        self.available = all([
            self.SMTP_SERVER,
            self.SMTP_PORT,
            self.EMAIL_USER,
            self.EMAIL_PASSWORD
        ])

EMAIL_CONFIG = EmailConfig()

def validate_email(email):
    """Valida el formato de un email"""
    if not email or email == "":
        return False
    if email == "usuario@ejemplo.com":
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def enviar_correo(destinatario, asunto, mensaje_html=None, mensaje_texto=None, archivos=None):
    """
    Envía correo electrónico con formato HTML y texto plano
    
    Args:
        destinatario: Email del destinatario
        asunto: Asunto del correo
        mensaje_html: Mensaje en formato HTML
        mensaje_texto: Mensaje en texto plano (alternativa)
        archivos: Lista de diccionarios con {'nombre': 'archivo.pdf', 'contenido': bytes}
    
    Returns:
        bool: True si se envió correctamente
    """
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
        
        # Adjuntar versión texto plano (si existe)
        if mensaje_texto:
            msg.attach(MIMEText(mensaje_texto, 'plain'))
        
        # Adjuntar versión HTML (si existe)
        if mensaje_html:
            msg.attach(MIMEText(mensaje_html, 'html'))
        
        # Adjuntar archivos
        if archivos:
            for archivo in archivos:
                if len(archivo['contenido']) > EMAIL_CONFIG.MAX_FILE_SIZE_MB * 1024 * 1024:
                    st.warning(f"El archivo {archivo['nombre']} excede el tamaño máximo de {EMAIL_CONFIG.MAX_FILE_SIZE_MB}MB y no será enviado")
                    continue
                
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(archivo['contenido'])
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename="{archivo["nombre"]}"')
                msg.attach(part)
        
        # Enviar correo
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
    """Configura NLTK y descarga los recursos necesarios"""
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
            st.warning("No se pudieron descargar stopwords. Usando lista básica.")
            return False
    
    try:
        from nltk.corpus import stopwords
        from nltk.stem import SnowballStemmer
        return True
    except:
        return False

# Configurar NLTK al inicio
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
    .badge-semantic { 
        background-color: #E53935; 
        color: white; 
        padding: 0.2rem 0.8rem; 
        border-radius: 15px; 
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .badge-doaj { 
        background-color: #9C27B0; 
        color: white; 
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
    .section-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .section-abstract { background-color: #e1f5fe; color: #01579b; }
    .section-introduction { background-color: #fff3e0; color: #bf360c; }
    .section-methods { background-color: #e8f5e8; color: #1b5e20; }
    .section-results { background-color: #f3e5f5; color: #4a148c; }
    .section-discussion { background-color: #fff8e1; color: #ff6f00; }
    .section-conclusion { background-color: #ffebee; color: #b71c1c; }
    
    .progress-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stats-box {
        background-color: #e8f5e9;
        border-left: 5px solid #2e7d32;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .evidence-box {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 0.5rem;
        margin: 0.2rem 0;
        font-size: 0.85rem;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .assistant-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .assistant-step {
        background-color: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
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
    .email-box {
        background-color: #fce4e4;
        border-left: 5px solid #e53935;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .strong-correlation {
        background: linear-gradient(135deg, #1E88E5, #0d47a1);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 0.5rem;
    }
    .reference-card {
        background-color: #e8f0fe;
        border-left: 5px solid #1E88E5;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .reference-title {
        font-weight: bold;
        color: #0d47a1;
    }
    .reference-authors {
        color: #424242;
        font-style: italic;
    }
    .reference-journal {
        color: #1E88E5;
    }
    .correlation-badge {
        background-color: #4CAF50;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# TOKENIZADOR ALTERNATIVO
# ============================================================================

def simple_sent_tokenize(text: str) -> List[str]:
    """Tokenizador simple de oraciones como fallback"""
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
    """Maneja traducciones con caché"""
    
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
        
        # Ejemplos predefinidos
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
                "name": "COVID-19 y daño miocárdico",
                "sujeto": "infección por COVID-19",
                "efecto": "daño miocárdico",
                "poblacion": "pacientes hospitalizados",
                "tipo": "causal",
                "verbo": "causa",
                "hypothesis": "La infección por COVID-19 causa daño miocárdico en pacientes hospitalizados"
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
        """Obtiene verbos disponibles para un tipo de plantilla"""
        if template_type in self.templates:
            return self.templates[template_type].get(f"verbs_{language}", [])
        return []
    
    def build_hypothesis(self, subject: str, effect: str, population: str, 
                        template_type: str, verb: str = None) -> Dict:
        """
        Construye una conjetura bien formada en español e inglés
        
        Args:
            subject: Sujeto de estudio (ej: "ticagrelor", "ejercicio")
            effect: Efecto observado (ej: "disnea", "mejoría")
            population: Población de estudio
            template_type: Tipo de plantilla
            verb: Verbo específico (opcional)
        
        Returns:
            Dict con conjeturas en español e inglés
        """
        if template_type not in self.templates:
            return {"es": "", "en": ""}
        
        template = self.templates[template_type]
        
        # Si no se especifica verbo, usar el primero de la lista
        if not verb and template.get(f"verbs_es"):
            verb = template[f"verbs_es"][0]
        
        # Construir en español
        if verb:
            # Para plantillas que tienen el verbo como parte de la frase
            if "{verbo}" in template["es"]:
                hypothesis_es = template["es"].format(
                    sujeto=subject,
                    verbo=verb,
                    efecto=effect,
                    poblacion=population
                )
            else:
                # Para plantillas que ya incluyen el verbo en la estructura
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
        
        # Traducir al inglés
        hypothesis_en = self.translator.translate_to_english(hypothesis_es)
        
        # Construcción directa en inglés para mayor precisión
        subject_en = self.translator.translate_to_english(subject)
        effect_en = self.translator.translate_to_english(effect)
        population_en = self.translator.translate_to_english(population)
        
        # Encontrar verbo en inglés correspondiente
        verb_en = ""
        if verb and template.get(f"verbs_es") and template.get(f"verbs_en"):
            try:
                verb_idx = template["verbs_es"].index(verb)
                verbs_en = template["verbs_en"]
                verb_en = verbs_en[verb_idx] if verb_idx < len(verbs_en) else verbs_en[0]
            except ValueError:
                # Si el verbo no está en la lista, usar traducción
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
        """Renderiza la interfaz del asistente en Streamlit"""
        
        with st.expander("🤖 ASISTENTE DE CONJETURAS - Ayuda a construir tu hipótesis", expanded=False):
            st.markdown('<div class="assistant-box">', unsafe_allow_html=True)
            st.markdown("### 🎯 Construye tu conjetura científica")
            st.markdown("Completa los siguientes campos para generar una hipótesis bien formada:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Selección de ejemplo o entrada manual
                example_option = st.selectbox(
                    "📋 Cargar ejemplo:",
                    ["Personalizado"] + [e["name"] for e in self.examples],
                    key="example_selector"
                )
                
                # Campos del formulario
                subject = st.text_input(
                    "🧪 Sujeto/Intervención:",
                    value="",
                    placeholder="Ej: ticagrelor, ejercicio, vacuna, fármaco X",
                    help="¿Qué elemento estás estudiando?"
                )
                
                effect = st.text_input(
                    "📊 Efecto/Desenlace:",
                    value="",
                    placeholder="Ej: disnea, mejoría, mortalidad, efecto secundario",
                    help="¿Qué efecto esperas observar?"
                )
                
                population = st.text_input(
                    "👥 Población:",
                    value="",
                    placeholder="Ej: pacientes con cardiopatía, adultos mayores, niños",
                    help="¿En qué población?"
                )
            
            with col2:
                # Tipo de relación
                template_type = st.selectbox(
                    "🔄 Tipo de relación:",
                    options=list(self.templates.keys()),
                    format_func=lambda x: f"{x} - {self.templates[x]['description']}",
                    key="template_type"
                )
                
                # Verbos disponibles según el tipo
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
            
            # Botón para generar
            if st.button("✨ GENERAR CONJETURA", type="primary", use_container_width=True):
                if subject and effect and population:
                    # Construir hipótesis
                    hypothesis_data = self.build_hypothesis(
                        subject=subject,
                        effect=effect,
                        population=population,
                        template_type=template_type,
                        verb=verb
                    )
                    
                    # Mostrar resultado
                    st.markdown("---")
                    st.markdown("### 📝 CONJETURA GENERADA")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🇪🇸 Español:**")
                        st.success(hypothesis_data["es"])
                        
                        # Botón para usar esta hipótesis
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
                    
                    # Guardar en session state
                    st.session_state['last_hypothesis_data'] = hypothesis_data
                    
                else:
                    st.warning("⚠️ Completa todos los campos para generar la conjetura")
            
            # Cargar ejemplo si se selecciona
            if example_option != "Personalizado":
                example = next((e for e in self.examples if e["name"] == example_option), None)
                if example:
                    st.markdown("---")
                    st.markdown("### 📋 Ejemplo cargado:")
                    st.info(f"**Hipótesis:** {example['hypothesis']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📌 Usar este ejemplo", key="use_example"):
                            st.session_state['hypothesis'] = example['hypothesis']
                            st.session_state['query'] = f'("{example["sujeto"]}" AND "{example["efecto"]}")'
                            # También guardar traducción aproximada
                            translator = TranslationManager()
                            st.session_state['hypothesis_en'] = translator.translate_to_english(example['hypothesis'])
                            st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Consejos adicionales
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
        """
        Traduce una hipótesis en español a inglés para búsqueda
        """
        if not hypothesis_es:
            return ""
        
        # Si ya tenemos una traducción en session state, usarla
        if 'hypothesis_en' in st.session_state and st.session_state['hypothesis_en']:
            return st.session_state['hypothesis_en']
        
        # Traducir
        return self.translator.translate_to_english(hypothesis_es)


# ============================================================================
# VERIFICADOR SEMÁNTICO AVANZADO CON IA
# ============================================================================

class AdvancedSemanticVerifier:
    """
    Verificador semántico avanzado con técnicas de IA modernas
    - Análisis semántico profundo
    - Detección de matices lingüísticos
    - Evaluación de calidad metodológica
    - Ponderación por secciones
    """
    
    def __init__(self):
        # Configurar NLTK si está disponible
        if NLTK_READY:
            self.stop_words_es = set(stopwords.words('spanish'))
            self.stop_words_en = set(stopwords.words('english'))
            self.stemmer = SnowballStemmer('spanish')
        else:
            # Stop words básicas como fallback
            self.stop_words_es = {'el', 'la', 'los', 'las', 'de', 'del', 'y', 'o', 
                                  'a', 'en', 'por', 'para', 'con', 'sin', 'sobre'}
            self.stop_words_en = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on',
                                 'at', 'to', 'for', 'with', 'without', 'by'}
            self.stemmer = None
        
        self.translator = TranslationManager()
        
        # ====================================================================
        # PESOS POR SECCIÓN DEL ARTÍCULO (basado en importancia científica)
        # ====================================================================
        self.section_weights = {
            'abstract': 1.5,
            'introduction': 0.6,
            'methods': 0.4,
            'results': 1.8,
            'discussion': 1.5,
            'conclusion': 1.8,
            'unknown': 1.0
        }
        
        # ====================================================================
        # KEYWORDS PARA DETECTAR SECCIONES
        # ====================================================================
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
        
        # ====================================================================
        # DETECTOR DE MATICES LINGÜÍSTICOS (NIVELES DE CERTEZA)
        # ====================================================================
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
        
        # ====================================================================
        # PATRONES DE NEGACIÓN
        # ====================================================================
        self.negation_patterns = [
            r'no\s+(hay|existe|se\s+observó|se\s+encontró)\s+(asociación|relación|correlación)',
            r'no\s+(se\s+)?(demostró|evidenció|confirmó)\s+',
            r'no\s+(fue|resultó)\s+(significativo|estadísticamente\s+significativo)',
            r'p\s*[>=]\s*0\.0[5-9]',
            r'p\s*>\s*0\.05',
            r'not\s+(associated|related|correlated)',
            r'no\s+(significant|statistically\s+significant)'
        ]
        
        # ====================================================================
        # TIPOS DE RELACIÓN (con pesos)
        # ====================================================================
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
        
        # ====================================================================
        # EVALUACIÓN DE CALIDAD DE ESTUDIOS
        # ====================================================================
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
        
        # ====================================================================
        # PESOS FINALES PARA VOTACIÓN
        # ====================================================================
        self.vote_weights = {
            'causal': 2.2,
            'risk': 2.0,
            'association': 1.8,
            'protective': 1.5,
            'statistical': 1.8,
            'moderate': 1.0,
            'strong_negation': -1.5,
            'statistical_negation': -1.8
        }
        
        self.UMBRAL_RELEVANCIA = 0.1
    
    # ========================================================================
    # MÉTODOS DE ANÁLISIS DE TEXTO
    # ========================================================================
    
    def detect_section(self, text_block: str) -> str:
        """Detecta la sección del artículo basado en keywords"""
        text_lower = text_block.lower()[:500]
        
        for section, keywords in self.section_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return section
        
        return 'unknown'
    
    def extract_key_terms(self, hypothesis: str) -> List[str]:
        """Extrae términos clave de la hipótesis con expansión semántica"""
        words = re.findall(r'\b[a-zA-Záéíóúñü]+\b', hypothesis.lower())
        
        filtered_words = []
        for word in words:
            if len(word) > 3 and word not in self.stop_words_es and word not in self.stop_words_en:
                if self.stemmer:
                    word = self.stemmer.stem(word)
                filtered_words.append(word)
        
        # Añadir bigramas relevantes (frases de dos palabras)
        bigrams = re.findall(r'\b[a-zA-Záéíóúñü]+\s+[a-zA-Záéíóúñü]+\b', hypothesis.lower())
        for bigram in bigrams:
            if all(len(w) > 3 for w in bigram.split()):
                filtered_words.append(bigram.replace(' ', '_'))
        
        return list(set(filtered_words))
    
    def analyze_sentence_deep(self, sentence: str) -> Dict:
        """
        Análisis profundo de una oración con detección de:
        - Tipo de relación
        - Nivel de certeza
        - Negación
        - Fuerza de la evidencia
        """
        sentence_lower = sentence.lower()
        
        # Detectar negación
        has_negation = False
        for pattern in self.negation_patterns:
            if re.search(pattern, sentence_lower, re.IGNORECASE):
                has_negation = True
                break
        
        # Detectar nivel de certeza
        certainty = 'unknown'
        certainty_weight = 0.5
        for level, data in self.certainty_levels.items():
            if any(term in sentence_lower for term in data['terms']):
                certainty = level
                certainty_weight = data['weight']
                break
        
        # Detectar tipo de relación
        relation = 'unknown'
        relation_weight = 0.5
        for rel_type, data in self.relation_types.items():
            if any(term in sentence_lower for term in data['terms']):
                relation = rel_type
                relation_weight = data['weight']
                break
        
        # Calcular fuerza de la evidencia
        strength = relation_weight * certainty_weight
        if has_negation:
            strength *= -1  # La negación invierte la dirección
        
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
        """Evalúa la calidad metodológica del estudio"""
        text_lower = text.lower()
        
        # Detectar tipo de estudio
        study_type = 'unknown'
        study_weight = 1.0
        for stype, patterns in self.study_type_patterns.items():
            if any(p in text_lower for p in patterns):
                study_type = stype
                study_weight = self.study_type_weights.get(stype, 1.0)
                break
        
        # Detectar tamaño de muestra
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
        
        # Detectar factores de calidad adicionales
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
        """Calcula relevancia de la oración respecto a la hipótesis usando TF-IDF"""
        if not hypothesis_terms:
            return 0.0
        
        sentence_lower = sentence.lower()
        
        # Coincidencia exacta de términos
        matches = sum(1 for term in hypothesis_terms if term in sentence_lower)
        
        # Coincidencia de raíces (stemming)
        if self.stemmer:
            sentence_stems = [self.stemmer.stem(w) for w in sentence_lower.split()]
            hypothesis_stems = [self.stemmer.stem(term.replace('_', ' ')) for term in hypothesis_terms]
            stem_matches = sum(1 for stem in hypothesis_stems if any(stem in s for s in sentence_stems))
            matches = max(matches, stem_matches)
        
        score = matches / len(hypothesis_terms)
        
        # Bonus por proximidad de términos
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
        """Divide texto en oraciones"""
        if NLTK_READY:
            try:
                return sent_tokenize(text)
            except:
                return simple_sent_tokenize(text)
        else:
            return simple_sent_tokenize(text)
    
    # ========================================================================
    # MÉTODO PRINCIPAL DE VERIFICACIÓN
    # ========================================================================
    
    def verify_article_text(self, text: str, hypothesis: str) -> Dict:
        """
        Verifica un artículo completo usando técnicas avanzadas de IA
        
        Args:
            text: Texto completo del artículo
            hypothesis: Hipótesis a verificar (en inglés)
        
        Returns:
            Dict con resultados del análisis
        """
        if not text or len(text.strip()) < 100:
            return {
                'success': False,
                'error': 'Texto insuficiente para análisis',
                'verdict': None
            }
        
        # PASO 1: Extraer términos clave de la hipótesis
        hypothesis_terms = self.extract_key_terms(hypothesis)
        
        # PASO 2: Evaluar calidad del estudio
        quality = self.assess_study_quality(text)
        
        # PASO 3: Dividir en oraciones
        sentences = self.split_sentences(text)
        
        # PASO 4: Dividir en bloques para detección de secciones
        block_size = max(1, len(text) // 10)
        blocks = [text[i:i+block_size] for i in range(0, len(text), block_size)]
        
        # Mapa de secciones
        section_map = {}
        current_section = 'unknown'
        for i, block in enumerate(blocks):
            detected = self.detect_section(block)
            if detected != 'unknown':
                current_section = detected
            section_map[i] = current_section
        
        # PASO 5: Analizar cada oración
        evidence_list = []
        section_counts = Counter()
        
        for i, sentence in enumerate(sentences):
            # Determinar sección
            block_idx = min(i // max(1, len(sentences) // len(blocks)), len(blocks)-1)
            section = section_map.get(block_idx, 'unknown')
            section_weight = self.section_weights.get(section, 1.0)
            
            # Calcular relevancia
            relevance = self.calculate_relevance(sentence, hypothesis_terms)
            
            if relevance > self.UMBRAL_RELEVANCIA:
                # Traducir para mejor análisis
                sentence_en = self.translator.translate_to_english(sentence)
                
                # Análisis profundo de la oración
                analysis = self.analyze_sentence_deep(sentence_en)
                
                if analysis['direction'] != 0:  # Si hay evidencia (positiva o negativa)
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
        
        # PASO 6: Votación ponderada
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
        """
        Sistema de votación ponderada con factores:
        - Tipo de relación
        - Nivel de certeza
        - Sección del artículo
        - Calidad del estudio
        - Relevancia
        """
        if not evidence_list:
            return {
                'score': 0,
                'confidence': 0,
                'verdict': 'inconclusive',
                'verdict_text': 'EVIDENCIA NO CONCLUYENTE',
                'support_count': 0,
                'against_count': 0
            }
        
        # Pesos base por tipo de relación
        relation_weights = {
            'causal': 2.2,
            'risk': 2.0,
            'association': 1.8,
            'protective': 1.5,
            'unknown': 1.0
        }
        
        # Pesos por nivel de certeza
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
            # Peso base por relación
            base_weight = relation_weights.get(evidence['relation'], 1.0)
            
            # Peso por certeza
            certainty_weight = certainty_weights.get(evidence['certainty'], 0.5)
            
            # Peso por sección
            section_weight = evidence['section_weight']
            
            # Peso por relevancia
            relevance_bonus = 1 + evidence['relevance'] * 0.5
            
            # Peso por calidad del estudio
            quality_multiplier = quality['type_weight'] if quality else 1.0
            
            # Peso final
            final_weight = (base_weight * certainty_weight * section_weight * 
                          relevance_bonus * quality_multiplier)
            
            # Aplicar dirección (positiva o negativa)
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
        
        # Normalizar confianza basado en cantidad y calidad de evidencia
        evidence_quality = len(evidence_list) / 20.0
        strong_ratio = len(strong_evidence) / max(1, len(evidence_list))
        norm_confidence = min(1.0, (evidence_quality + strong_ratio) / 1.5)
        
        # Determinar veredicto con umbrales ajustados
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
    """
    Motor de búsqueda científica mejorado para grandes volúmenes de resultados
    """
    
    def __init__(self, email: str):
        self.email = email
        self.delay = 0.3
        
    def format_pubmed_query(self, query: str, year_range: tuple = None) -> str:
        """Formatea consultas complejas de PubMed incluyendo rango de años"""
        query = ' '.join(query.split())
        
        mesh_pattern = r'"([^"]+)"\[Mesh\]'
        query = re.sub(mesh_pattern, r'\1[mh]', query)
        
        pubtype_pattern = r'"([^"]+)"\[Publication Type\]'
        query = re.sub(pubtype_pattern, r'\1[pt]', query)
        
        field_mappings = {
            '[Mesh]': '[mh]',
            '[Publication Type]': '[pt]',
            '[Title/Abstract]': '[tiab]',
            '[Author]': '[au]',
            '[Journal]': '[ta]'
        }
        
        for old, new in field_mappings.items():
            query = query.replace(old, new)
        
        if year_range and year_range[0] and year_range[1]:
            year_filter = f" AND ({year_range[0]}[pdat] : {year_range[1]}[pdat])"
            query = f"({query}){year_filter}"
        
        return query
    
    def fetch_pubmed_batch(self, ids_batch: list) -> list:
        """Obtiene detalles de un lote de IDs de PubMed"""
        results = []
        try:
            ids = ','.join(ids_batch)
            fetch_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                f"?db=pubmed"
                f"&id={ids}"
                f"&retmode=xml"
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
                    title = title_elem.text if title_elem is not None else "Título no disponible"
                    
                    authors = []
                    author_list = article.findall('.//Author')
                    for author in author_list[:5]:
                        last = author.find('LastName')
                        fore = author.find('ForeName')
                        if last is not None and fore is not None:
                            authors.append(f"{fore.text} {last.text}")
                        elif last is not None:
                            authors.append(last.text)
                    
                    journal_elem = article.find('.//Title')
                    journal = journal_elem.text if journal_elem is not None else ""
                    
                    year_elem = article.find('.//PubDate/Year')
                    if year_elem is None:
                        year_elem = article.find('.//PubDate/MedlineDate')
                    year = year_elem.text[:4] if year_elem is not None and year_elem.text else ""
                    
                    doi_elem = article.find(".//ArticleId[@IdType='doi']")
                    doi = doi_elem.text if doi_elem is not None else ""
                    
                    pmid_elem = article.find(".//PMID")
                    pmid = pmid_elem.text if pmid_elem is not None else ""
                    
                    abstract_elem = article.find('.//AbstractText')
                    abstract = abstract_elem.text if abstract_elem is not None else ""
                    
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
            pass
        
        return results
    
    def search_pubmed_advanced(self, query: str, max_results: int = 1000, year_range: tuple = None) -> list:
        """Búsqueda avanzada en PubMed con soporte para hasta 1000 resultados"""
        all_results = []
        try:
            formatted_query = self.format_pubmed_query(query, year_range)
            encoded_query = urllib.parse.quote(formatted_query)
            
            # Obtener IDs
            search_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                f"?db=pubmed"
                f"&term={encoded_query}"
                f"&retmode=json"
                f"&retmax={min(1000, max_results)}"
                f"&sort=relevance"
                f"&tool=streamlit_app"
                f"&email={self.email}"
            )
            
            time.sleep(self.delay)
            response = requests.get(search_url, timeout=30)
            response.raise_for_status()
            search_data = response.json()
            
            id_list = search_data.get('esearchresult', {}).get('idlist', [])
            
            if id_list:
                # Obtener detalles en lotes
                batch_size = 50
                id_batches = [id_list[i:i + batch_size] for i in range(0, len(id_list), batch_size)]
                
                for batch in id_batches:
                    batch_results = self.fetch_pubmed_batch(batch)
                    all_results.extend(batch_results)
                    
        except Exception as e:
            st.warning(f"Error en PubMed: {str(e)}")
        
        return all_results
    
    def search_crossref(self, query: str, max_results: int = 1000, year_range: tuple = None) -> list:
        """Busca en CrossRef con soporte para hasta 1000 resultados"""
        results = []
        try:
            # Ya no limpiamos la query aquí, recibimos una versión ya simplificada
            url = "https://api.crossref.org/works"
            params = {
                'query': query[:200],
                'rows': min(100, max_results),
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
        
        return results
    
    def search_openalex(self, query: str, max_results: int = 1000, year_range: tuple = None) -> list:
        """Busca en OpenAlex con soporte para hasta 1000 resultados"""
        results = []
        try:
            # Ya no limpiamos la query aquí, recibimos una versión ya simplificada
            url = "https://api.openalex.org/works"
            params = {
                'search': query[:200],
                'per-page': 50,
                'sort': 'relevance_score:desc'
            }
            
            if year_range and year_range[0] and year_range[1]:
                params['filter'] = f"publication_year:{year_range[0]}-{year_range[1]}"
            
            time.sleep(self.delay)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for item in data.get('results', []):
                results.append({
                    'base_datos': 'OpenAlex',
                    'titulo': item.get('title', 'Título no disponible'),
                    'autores': ', '.join([a.get('author', {}).get('display_name', '') 
                                         for a in item.get('authorships', [])[:5]]),
                    'revista': item.get('host_venue', {}).get('display_name', '') if item.get('host_venue') else '',
                    'año': str(item.get('publication_year', '')),
                    'doi': item.get('doi', '').replace('https://doi.org/', ''),
                    'url': item.get('doi', ''),
                    'tipo': item.get('type', ''),
                    'open_access': item.get('open_access', {}).get('oa_url', '')
                })
                
        except Exception as e:
            st.warning(f"Error en OpenAlex: {str(e)}")
        
        return results[:max_results]
    
    def search_europe_pmc(self, query: str, max_results: int = 1000, year_range: tuple = None) -> list:
        """Busca en Europe PMC con soporte para hasta 1000 resultados"""
        results = []
        try:
            # Ya no limpiamos la query aquí, recibimos una versión ya simplificada
            if year_range and year_range[0] and year_range[1]:
                date_filter = f" AND (PUB_YEAR:{year_range[0]}-{year_range[1]})"
                query = f"({query}){date_filter}"
            
            url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            params = {
                'query': query[:500],
                'format': 'json',
                'pageSize': min(100, max_results),
                'resultType': 'core'
            }
            
            time.sleep(self.delay)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for item in data.get('resultList', {}).get('result', []):
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
                
        except Exception as e:
            st.warning(f"Error en Europe PMC: {str(e)}")
        
        return results[:max_results]
    
    def search_all(self, query: str, max_results_per_db: int = 1000, selected_dbs: list = None, 
                   year_range: tuple = None) -> pd.DataFrame:
        """
        Busca en todas las bases de datos seleccionadas con alto volumen
        PubMed usa la consulta original con sintaxis MeSH
        Las otras bases usan una versión simplificada
        """
        
        if selected_dbs is None:
            selected_dbs = ['PubMed', 'CrossRef', 'OpenAlex', 'Europe PMC']
        
        all_results = []
        
        # Crear una copia para simplificar (NO modificar la original)
        simple_query = query
        
        # Crear una versión simplificada de la consulta para bases que no soportan MeSH
        # Eliminar tags MeSH y Publication Type pero mantener términos clave
        simple_query = re.sub(r'\[[^\]]*\]', '', simple_query)  # Elimina [Mesh], [PT], etc.
        simple_query = re.sub(r'"[^"]*"', lambda m: m.group(0).replace('"', ''), simple_query)  # Quita comillas pero mantiene términos
        simple_query = re.sub(r'\(|\)', ' ', simple_query)  # Reemplaza paréntesis por espacios
        simple_query = re.sub(r'\s+', ' ', simple_query).strip()  # Normaliza espacios
        
        # Si la consulta simplificada es muy larga, extraer solo términos clave
        if len(simple_query) > 200:
            # Buscar términos médicos clave
            important_terms = re.findall(r'\b(ticagrelor|myocardial|ischemia|acute coronary|angina|infarction|cohort|prospective|trial|study|adults|adult|risk|factor|dyspnea|outcome|effect|treatment|therapy|patient|clinical|randomized|controlled)\b', 
                                       simple_query, re.IGNORECASE)
            if important_terms:
                # Tomar términos únicos (máximo 6)
                simple_query = ' '.join(list(set(important_terms))[:6])
            else:
                # Si no encuentra términos clave, tomar las primeras palabras
                words = simple_query.split()[:10]
                simple_query = ' '.join(words)
        
        search_functions = {
            'PubMed': self.search_pubmed_advanced,      # PubMed usa query original
            'CrossRef': self.search_crossref,           # CrossRef usa simplificada
            'OpenAlex': self.search_openalex,           # OpenAlex usa simplificada
            'Europe PMC': self.search_europe_pmc        # Europe PMC usa simplificada
        }
        
        for db in selected_dbs:
            if db in search_functions:
                try:
                    with st.spinner(f"🔍 Buscando en {db}..."):
                        # Para PubMed usar query original, para otras usar simplificada
                        if db == 'PubMed':
                            results = search_functions[db](query, max_results_per_db, year_range)
                        else:
                            results = search_functions[db](simple_query, max_results_per_db, year_range)
                        all_results.extend(results)
                        if results:
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
# OBTENEDOR DE TEXTO DE ARTÍCULOS (MEJORADO)
# ============================================================================

class ArticleTextFetcher:
    """Obtiene el texto completo de artículos desde fuentes abiertas"""
    
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
        """Control de tasa simple"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_delay:
            time.sleep(self.min_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    def get_text_from_doi(self, doi: str) -> Tuple[Optional[str], str]:
        """
        Obtiene el texto completo del artículo usando el DOI
        Retorna (texto, fuente)
        """
        if not doi or pd.isna(doi) or doi == '':
            return None, "DOI vacío"
        
        doi = str(doi).strip()
        if doi.startswith('https://doi.org/'):
            doi = doi.replace('https://doi.org/', '')
        elif doi.startswith('doi:'):
            doi = doi.replace('doi:', '')
        
        self.wait()
        
        # Intentar 1: OpenAlex (para obtener URLs Open Access)
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
        
        # Intentar 2: Europe PMC (tienen abstracts y metadata)
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
        
        # Intentar 3: Unpaywall
        try:
            url = f"https://api.unpaywall.org/v2/{doi}?email=usuario@ejemplo.com"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('is_oa') and data.get('best_oa_location'):
                    pdf_url = data['best_oa_location'].get('url_for_pdf')
                    if pdf_url:
                        return f"PDF Open Access en: {pdf_url}", "Unpaywall"
        except:
            pass
        
        # Intentar 4: PubMed (si es artículo de PubMed)
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
# CLASE PRINCIPAL INTEGRADA (ALTO VOLUMEN)
# ============================================================================

class IntegratedScientificVerifier:
    """
    Clase principal que integra búsqueda y verificación semántica
    Soporta hasta 1000 artículos por base de datos
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
            'inconclusos': 0,
            'strong_correlation': 0
        }
    
    def run_analysis(self, query: str, hypothesis: str, max_results_per_db: int = 1000, 
                     selected_dbs: list = None, year_range: tuple = None,
                     progress_callback=None) -> pd.DataFrame:
        """
        Ejecuta el flujo completo: búsqueda + análisis semántico
        """
        self.results = []
        self.stats = {k: 0 for k in self.stats}
        
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
            progress_callback(f"✅ Encontrados {len(articles_df)} artículos. Iniciando análisis...", 0.1)
        
        # PASO 2: ANÁLISIS DE CADA ARTÍCULO
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
                'oraciones_totales': 0,
                'oraciones_relevantes': 0,
                'detalle_evidencia': ''
            }
            
            if article_text:
                self.stats['with_text'] += 1
                
                # Analizar semánticamente
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
                    
                    # Actualizar estadísticas
                    self.stats['analyzed'] += 1
                    if 'CORROBORA' in verdict['verdict_text']:
                        self.stats['corroboran'] += 1
                        if verdict['confidence'] > 0.7 or verdict['score'] > 0.6:
                            self.stats['strong_correlation'] += 1
                    elif 'CONTRADICE' in verdict['verdict_text']:
                        self.stats['contradicen'] += 1
                    else:
                        self.stats['inconclusos'] += 1
                    
                    # Guardar evidencia resumida
                    if analysis['evidence']:
                        ev_summary = []
                        for ev in analysis['evidence'][:3]:
                            ev_summary.append(f"[{ev['section']}] {ev['relation']} ({ev['certainty']})")
                        result_row['detalle_evidencia'] = ' | '.join(ev_summary)
            else:
                result_row['veredicto'] = 'TEXTO NO DISPONIBLE'
            
            results_list.append(result_row)
            
            # Pequeña pausa entre análisis
            time.sleep(0.1)
        
        if progress_callback:
            progress_callback("✅ Análisis completado", 1.0)
        
        self.results = pd.DataFrame(results_list)
        return self.results
    
    def get_strong_correlation_articles(self) -> pd.DataFrame:
        """
        Retorna los artículos que tienen una correlación fuerte con la hipótesis
        """
        if self.results.empty:
            return pd.DataFrame()
        
        # Filtrar artículos con veredicto de corroboración fuerte
        # o con alta confianza (>0.7) o puntuación alta (>0.6)
        strong_df = self.results[
            (self.results['veredicto'].str.contains('FUERTEMENTE', na=False)) |
            (self.results['confianza'] > 0.7) |
            (self.results['puntuacion'] > 0.6)
        ].copy()
        
        # Ordenar por confianza descendente
        if not strong_df.empty:
            strong_df = strong_df.sort_values('confianza', ascending=False)
        
        return strong_df
    
    def generate_report(self) -> str:
        """Genera un reporte textual de los resultados"""
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
        report.append(f"   ├─ Correlación fuerte: {self.stats['strong_correlation']}")
        report.append(f"❌ Contradicen: {self.stats['contradicen']}")
        report.append(f"⚠️ Inconclusos: {self.stats['inconclusos']}")
        report.append("")
        
        # Artículos con correlación fuerte
        strong_df = self.get_strong_correlation_articles()
        if not strong_df.empty:
            report.append("="*80)
            report.append("📈 ARTÍCULOS CON CORRELACIÓN FUERTE CON LA HIPÓTESIS")
            report.append("="*80)
            
            for idx, row in strong_df.iterrows():
                report.append(f"\n📄 {row['titulo']}")
                report.append(f"   Autores: {row['autores']}")
                report.append(f"   Revista: {row['revista']} | Año: {row['año']}")
                report.append(f"   Base: {row['base_datos']} | DOI: {row['doi']}")
                report.append(f"   Veredicto: {row['veredicto']} (Confianza: {row['confianza']:.1%})")
                report.append(f"   Evidencia: {row['evidencia_a_favor']} a favor, {row['evidencia_en_contra']} en contra")
                if row['detalle_evidencia']:
                    report.append(f"   Evidencia: {row['detalle_evidencia']}")
                report.append(f"   URL: {row['url']}")
        
        report.append("")
        report.append("="*80)
        report.append("DETALLE COMPLETO POR ARTÍCULO:")
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
        """Genera un reporte HTML de los resultados para enviar por correo"""
        if self.results.empty:
            return "<p>No hay resultados para generar reporte.</p>"
        
        strong_df = self.get_strong_correlation_articles()
        
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
                .strong-correlation {{ background-color: #e8f0fe; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .reference {{ background-color: #e8f0fe; border-left: 5px solid #1E88E5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .reference-title {{ font-weight: bold; color: #0d47a1; }}
                .reference-authors {{ color: #424242; font-style: italic; }}
                .reference-journal {{ color: #1E88E5; }}
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
                <p><strong>   └─ Correlación fuerte:</strong> {self.stats['strong_correlation']}</p>
                <p><strong>❌ Contradicen:</strong> {self.stats['contradicen']}</p>
                <p><strong>⚠️ Inconclusos:</strong> {self.stats['inconclusos']}</p>
            </div>
        """
        
        if not strong_df.empty:
            html += f"""
            <h2>📈 Artículos con Correlación Fuerte</h2>
            <div class="strong-correlation">
                <p><strong>Se encontraron {len(strong_df)} artículos con correlación fuerte a la hipótesis:</strong></p>
            """
            
            for idx, row in strong_df.iterrows():
                html += f"""
                <div class="reference">
                    <p class="reference-title">📄 {row['titulo']}</p>
                    <p class="reference-authors">👥 {row['autores']}</p>
                    <p class="reference-journal">📚 {row['revista']} ({row['año']}) | {row['base_datos']}</p>
                    <p>🔗 DOI: <a href="https://doi.org/{row['doi']}" target="_blank">{row['doi']}</a></p>
                    <p>⚖️ Veredicto: <span class="verdict-assert">{row['veredicto']}</span> (Confianza: {row['confianza']:.1%})</p>
                    <p>📊 Evidencia: {row['evidencia_a_favor']} a favor, {row['evidencia_en_contra']} en contra</p>
                    <p>🔬 URL: <a href="{row['url']}" target="_blank">Ver artículo</a></p>
                </div>
                """
            
            html += "</div>"
        
        html += f"""
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
            # Manejar valores nulos o vacíos
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
    """Devuelve la clase CSS para el badge de base de datos"""
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
    """Devuelve la clase CSS para el veredicto"""
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
    """Envía los resultados del análisis por correo electrónico"""
    
    if integrator is None or integrator.results is None or integrator.results.empty:
        st.warning("No hay resultados para enviar por correo.")
        return False
    
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
        
        # Añadir hoja de correlación fuerte
        strong_df = integrator.get_strong_correlation_articles()
        if not strong_df.empty:
            strong_df.to_excel(writer, sheet_name='Correlacion_Fuerte', index=False)
    
    archivos.append({
        'nombre': f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        'contenido': excel_buffer.getvalue()
    })
    
    # Asunto y mensaje
    strong_count = integrator.stats.get('strong_correlation', 0)
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
            <li><strong>   └─ Correlación fuerte:</strong> {strong_count}</li>
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
    -    └─ Correlación fuerte: {strong_count}
    - ❌ Contradicen: {integrator.stats['contradicen']}
    - ⚠️ Inconclusos: {integrator.stats['inconclusos']}
    
    Los archivos adjuntos contienen el detalle completo.
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
# INTERFAZ PRINCIPAL DE STREAMLIT
# ============================================================================

def main():
    """Función principal de la aplicación"""
    
    st.markdown('<h1 class="main-header">🔬 Buscador y Verificador Semántico Integrado</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ALTO VOLUMEN: Hasta 1000 artículos por base • Análisis AI automático</p>', 
                unsafe_allow_html=True)
    
    # Inicializar session state
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
    
    # Inicializar asistente de hipótesis
    hypothesis_assistant = HypothesisAssistant()
    
    # Renderizar asistente
    hypothesis_assistant.render_assistant_ui()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
        st.markdown("## ⚙️ Configuración")
        
        email = st.text_input(
            "📧 Email (requerido para NCBI y para recibir resultados)", 
            value=st.session_state.get('user_email', ''),
            placeholder="tu@email.com",
            help="Este email se usará para NCBI y para enviarte los resultados"
        )
        st.session_state['user_email'] = email
        
        # Estado de NLTK
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
            min_value=10, max_value=1000, value=50, step=10,
            help="Máximo de artículos por base de datos"
        )
        
        st.markdown("### 🔧 Umbrales de análisis")
        min_relevance = st.slider(
            "Relevancia mínima",
            min_value=0.05,
            max_value=0.3,
            value=0.1,
            step=0.05,
            help="Mínimo de coincidencia con términos de la hipótesis"
        )
        
        st.markdown("### ⚠️ Advertencia")
        st.warning("""
        **Tiempo de procesamiento:**
        - 50 artículos: ~2-3 minutos
        - 100 artículos: ~5-7 minutos
        - 200 artículos: ~10-15 minutos
        """)
        
        st.markdown("### 📋 Ejemplos")
        if st.button("Cargar ejemplo: Ticagrelor y disnea"):
            st.session_state['query'] = '((("Ticagrelor"[Mesh]) OR (ticagrelor)) AND ((((((((("Myocardial Ischemia"[Mesh]) OR ("Acute Coronary Syndrome"[Mesh])) OR ("Angina Pectoris"[Mesh])) OR ("Coronary Disease"[Mesh])) OR ("Coronary Artery Disease"[Mesh])) OR ("Kounis Syndrome"[Mesh])) OR ("Myocardial Infarction"[Mesh])) OR ("Myocardial Reperfusion Injury"[Mesh])) OR (((((((((MYOCARDIAL ISCHEMIA) OR (ACUTE CORONARY SYNDROME)) OR (ANGINA PECTORIS)) OR (CORONARY DISEASE)) OR (CORONARY ARTERY DISEASE)) OR (kounis syndrome)) OR (myocardial infarction)) OR (myocardial reperfusion injury)) OR (ischemic heart disease)))) AND ((((((cohort studies) OR (prospective studies)) OR ("prospective clinical trial")) OR ("clinical records")) OR (randomized clinical trial)) OR ((("Clinical Study" [Publication Type] OR "Observational Study" [Publication Type]) OR "Retrospective Studies"[Mesh]) OR "Randomized Controlled Trial" [Publication Type]))) AND (adults or adult)'
            st.session_state['hypothesis'] = "El ticagrelor causa disnea como efecto secundario en pacientes con cardiopatía isquémica"
            translator = TranslationManager()
            st.session_state['hypothesis_en'] = translator.translate_to_english(st.session_state['hypothesis'])
            st.rerun()
        
        if st.button("Cargar ejemplo: COVID-19"):
            st.session_state['query'] = '("COVID-19"[Mesh] AND "Myocardial Injury"[Mesh])'
            st.session_state['hypothesis'] = "La infección por COVID-19 causa daño miocárdico"
            translator = TranslationManager()
            st.session_state['hypothesis_en'] = translator.translate_to_english(st.session_state['hypothesis'])
            st.rerun()
    
    # Área principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_query = st.text_area(
            "🔍 Consulta de búsqueda:",
            value=st.session_state.get('query', ''),
            height=120,
            placeholder='Ej: (("Ticagrelor"[Mesh]) AND ("Myocardial Ischemia"[Mesh]) AND ("dyspnea"))',
            help="Puedes usar términos MeSH o palabras clave",
            key="search_query_input"
        )
        st.session_state['query'] = search_query
    
    with col2:
        hypothesis_es = st.text_area(
            "🔬 Conjetura a verificar (español):",
            value=st.session_state.get('hypothesis', ''),
            height=68,
            placeholder='Ej: El fármaco X causa efecto Y en pacientes con Z',
            help="Escribe tu hipótesis en español",
            key="hypothesis_input"
        )
        
        if hypothesis_es and hypothesis_es != st.session_state.get('hypothesis', ''):
            st.session_state['hypothesis'] = hypothesis_es
            translator = TranslationManager()
            st.session_state['hypothesis_en'] = translator.translate_to_english(hypothesis_es)
        
        # Mostrar traducción automática
        if st.session_state.get('hypothesis_en'):
            st.markdown(f"**🇬🇧 Inglés (para búsqueda):**")
            st.info(st.session_state['hypothesis_en'][:150] + ('...' if len(st.session_state['hypothesis_en']) > 150 else ''))
    
    col1, col2, col3 = st.columns(3)
    with col2:
        analyze_button = st.button("🚀 INICIAR ANÁLISIS INTEGRADO", type="primary", use_container_width=True)
    
    if analyze_button and search_query and hypothesis_es:
        user_email = st.session_state.get('user_email', '')
        if not user_email or not validate_email(user_email):
            st.error("❌ Ingresa un email válido en la barra lateral")
        else:
            # Obtener bases de datos seleccionadas
            selected_dbs = [db for db, selected in databases.items() if selected]
            
            if not selected_dbs:
                st.error("❌ Selecciona al menos una base de datos")
                return
            
            # Inicializar integrador
            integrator = IntegratedScientificVerifier(user_email)
            integrator.semantic_verifier.UMBRAL_RELEVANCIA = min_relevance
            
            # Usar la hipótesis en inglés para el análisis
            hypothesis_for_analysis = st.session_state.get('hypothesis_en', hypothesis_es)
            
            # Contenedores para progreso
            progress_container = st.container()
            with progress_container:
                st.markdown('<div class="progress-container">', unsafe_allow_html=True)
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_estimate = st.empty()
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Función de callback para actualizar progreso
            start_time = time.time()
            
            def update_progress(message, value):
                status_text.text(message)
                # Asegurar que value esté entre 0 y 1
                value = max(0.0, min(1.0, value))
                progress_bar.progress(value)
                elapsed = time.time() - start_time
                if value > 0:
                    estimated_total = elapsed / value
                    remaining = estimated_total * (1 - value)
                    if remaining > 0:
                        time_estimate.text(f"⏱️ Tiempo transcurrido: {elapsed:.1f}s | Estimado restante: {remaining:.1f}s")
            
            # Ejecutar análisis
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
            
            if not results_df.empty:
                st.success(f"✅ Análisis completado en {elapsed_time:.1f} segundos ({elapsed_time/60:.1f} minutos)")
                
                # Guardar integrador en session state
                st.session_state['integrator'] = integrator
                
                # ESTADÍSTICAS GLOBALES
                st.markdown("## 📊 RESULTADOS GLOBALES")
                
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.metric("Artículos encontrados", integrator.stats['total_articles'])
                with col2:
                    st.metric("Con texto", integrator.stats['with_text'])
                with col3:
                    st.metric("✅ Corroboran", integrator.stats['corroboran'])
                with col4:
                    st.metric("🔴 Correl. fuerte", integrator.stats['strong_correlation'])
                with col5:
                    st.metric("❌ Contradicen", integrator.stats['contradicen'])
                with col6:
                    st.metric("⚠️ Inconclusos", integrator.stats['inconclusos'])
                
                # GRÁFICOS
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribución por base de datos
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
                            'Semantic Scholar': '#E53935',
                            'DOAJ': '#9C27B0',
                            'Europe PMC': '#FF5722'
                        }
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Distribución de veredictos (solo analizados)
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
                
                # SECCIÓN: ARTÍCULOS CON CORRELACIÓN FUERTE
                st.markdown("## 📈 ARTÍCULOS CON CORRELACIÓN FUERTE CON LA HIPÓTESIS")
                st.markdown("Estos artículos presentan la evidencia más sólida a favor de tu hipótesis:")
                
                strong_df = integrator.get_strong_correlation_articles()
                
                if not strong_df.empty:
                    for idx, row in strong_df.iterrows():
                        badge_class = get_badge_class(row['base_datos'])
                        
                        st.markdown(f"""
                        <div class="reference-card">
                            <span class="{badge_class}">{row['base_datos']}</span>
                            <span class="correlation-badge">Correlación fuerte</span>
                            <div class="reference-title">{row['titulo']}</div>
                            <div class="reference-authors">{row['autores']}</div>
                            <div class="reference-journal">{row['revista']} ({row['año']})</div>
                            <div><b>DOI:</b> {row.get('doi', 'No disponible')}</div>
                            <div><b>Veredicto:</b> <span class="verdict-assert">{row['veredicto']}</span> (Confianza: {row['confianza']:.1%})</div>
                            <div><b>Evidencia:</b> {row['evidencia_a_favor']} a favor, {row['evidencia_en_contra']} en contra</div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns([1, 1, 4])
                        with col1:
                            if row.get('url') and pd.notna(row['url']):
                                st.link_button("🔗 Ver artículo", row['url'])
                        with col2:
                            if row.get('doi') and pd.notna(row['doi']):
                                doi_link = f"https://doi.org/{row['doi']}"
                                st.link_button("📋 DOI", doi_link)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.info("No se encontraron artículos con correlación fuerte. Revisa los artículos que corroboran para más detalles.")
                
                # TABLA DE RESULTADOS
                st.markdown("## 📋 ARTÍCULOS ANALIZADOS")
                
                # Filtrar para mostrar
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
                
                # DETALLE POR ARTÍCULO (limitado para no saturar)
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
                
                # REPORTE Y EXPORTACIÓN
                st.markdown("## 💾 EXPORTAR RESULTADOS")
                
                col1, col2, col3 = st.columns(3)
                
                # Reporte de texto
                report_text = integrator.generate_report()
                with col1:
                    st.download_button(
                        "📝 Reporte TXT",
                        report_text,
                        file_name=f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                # CSV
                with col2:
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📊 CSV",
                        csv,
                        file_name=f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                # Excel
                with col3:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        results_df.to_excel(writer, sheet_name='Resultados', index=False)
                        
                        summary = pd.DataFrame([integrator.stats])
                        summary.to_excel(writer, sheet_name='Resumen', index=False)
                        
                        if not results_df[results_df['veredicto'] != 'TEXTO NO DISPONIBLE'].empty:
                            verdict_stats = results_df[results_df['veredicto'] != 'TEXTO NO DISPONIBLE']['veredicto'].value_counts().reset_index()
                            verdict_stats.columns = ['Veredicto', 'Cantidad']
                            verdict_stats.to_excel(writer, sheet_name='Por Veredicto', index=False)
                        
                        # Añadir hoja de correlación fuerte
                        strong_df = integrator.get_strong_correlation_articles()
                        if not strong_df.empty:
                            strong_df.to_excel(writer, sheet_name='Correlacion_Fuerte', index=False)
                    
                    st.download_button(
                        "📥 Excel",
                        buffer.getvalue(),
                        file_name=f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            
            else:
                st.warning("😕 No se encontraron artículos. Prueba con otra consulta o amplía el rango de años.")
    
    # Sección para enviar resultados por correo (después del análisis)
    if st.session_state.get('integrator') is not None and not st.session_state.integrator.results.empty:
        st.markdown("---")
        st.markdown("## 📧 ENVIAR RESULTADOS POR CORREO")
        st.markdown('<div class="email-box">', unsafe_allow_html=True)
        
        strong_count = st.session_state.integrator.stats.get('strong_correlation', 0)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(f"📨 ENVIAR RESULTADOS A MI CORREO ({strong_count} correlaciones fuertes)", type="primary", use_container_width=True):
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
        <p>🔬 Buscador y Verificador Semántico Integrado v3.2 | ALTO VOLUMEN: Hasta 1000 artículos por base | Análisis AI avanzado | PubMed conserva sintaxis MeSH</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
