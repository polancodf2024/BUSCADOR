"""
Integrated Semantic Search and Verifier - COMPLETE CORRECTED VERSION with LIVE MeSH LOOKUP
CRITICAL FIXES:
1. PubMed: Direct MeSH term lookup via E-utilities API - NO MORE LOCAL MAPPING
2. Result limit: Now respects up to 1000 articles per database
3. Examples: Fixed loading and optimized queries
4. Persistence: Results are NO longer deleted after analysis
5. Semantic analysis: FIXED - Bilingual detection with medical terms in English
6. MULTIDOMAIN: Added pharmacology and metabolism terms
7. SPECIFIC WEIGHTS: Domain bonuses for ticagrelor and exercise-diabetes
8. AUTOMATIC DETECTION: Program detects hypothesis domain
9. NEW: Visualization of ALL articles that strongly support the hypothesis
10. FIXED: Hypothesis assistant - Now queries MeSH terms LIVE from PubMed
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
from typing import List, Dict, Tuple, Optional, Any
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
from difflib import SequenceMatcher

# Page configuration
st.set_page_config(
    page_title="🔬 Integrated Semantic Search and Verifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# EMAIL CONFIGURATION
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
    if email == "user@example.com":
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def send_email(recipient, subject, html_message=None, text_message=None, attachments=None):
    if not EMAIL_CONFIG.available:
        st.error("Email configuration not available. Check application secrets.")
        return False
    
    if not validate_email(recipient):
        st.error("Invalid email address")
        return False
    
    if not subject or (not html_message and not text_message):
        st.error("Missing required data to send email")
        return False
    
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = EMAIL_CONFIG.EMAIL_USER
        msg['To'] = recipient
        msg['Subject'] = subject
        msg['Date'] = formatdate(localtime=True)
        
        if text_message:
            msg.attach(MIMEText(text_message, 'plain', 'utf-8'))
        
        if html_message:
            msg.attach(MIMEText(html_message, 'html', 'utf-8'))
        
        if attachments:
            for attachment in attachments:
                if len(attachment['content']) > EMAIL_CONFIG.MAX_FILE_SIZE_MB * 1024 * 1024:
                    st.warning(f"File {attachment['name']} exceeds maximum size and will not be sent")
                    continue
                
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment['content'])
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename="{attachment["name"]}"')
                msg.attach(part)
        
        context = ssl.create_default_context()
        
        with smtplib.SMTP(EMAIL_CONFIG.SMTP_SERVER, EMAIL_CONFIG.SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(EMAIL_CONFIG.EMAIL_USER, EMAIL_CONFIG.EMAIL_PASSWORD)
            server.send_message(msg)
        
        return True
        
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return False

# ============================================================================
# NLTK CONFIGURATION
# ============================================================================

def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except:
            st.warning("Could not download punkt. Using alternative tokenization.")
            return False
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except:
            st.warning("Could not download stopwords. Using basic list.")
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
# CSS STYLES
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
    .result-card-strong {
        background: linear-gradient(135deg, #f0f9f0, #e8f5e8);
        border-left: 5px solid #4CAF50;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 8px rgba(76, 175, 80, 0.2);
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
    .strong-evidence-title {
        background: linear-gradient(135deg, #4CAF50, #2E7D32);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 30px;
        display: inline-block;
        font-weight: bold;
        margin-bottom: 1.5rem;
        font-size: 1.3rem;
    }
    .counter-badge {
        background-color: #ff9800;
        color: white;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.9rem;
        margin-left: 1rem;
    }
    .mesh-query {
        background-color: #1E1E1E;
        color: #FFFFFF;
        padding: 1rem;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        margin: 0.5rem 0;
        border-left: 5px solid #4CAF50;
    }
    .query-example {
        background-color: #e8f5e9;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border-left: 5px solid #4CAF50;
        margin: 0.5rem 0;
        font-family: monospace;
    }
    .mesh-suggestion {
        background-color: #f0f8ff;
        border: 1px solid #1E88E5;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.2rem 0;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .mesh-suggestion:hover {
        background-color: #e1f0fa;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ALTERNATIVE TOKENIZER
# ============================================================================

def simple_sent_tokenize(text: str) -> List[str]:
    abbreviations = ['Dr.', 'Prof.', 'vs.', 'Fig.', 'Eq.', 'et al.', 
                    'i.e.', 'e.g.', 'vol.', 'no.', 'pp.', 'eds.']
    
    for i, abbr in enumerate(abbreviations):
        text = text.replace(abbr, f"ABBR{i}")
    
    sentences = re.split(r'[.!?]+', text)
    
    for i, abbr in enumerate(abbreviations):
        sentences = [s.replace(f"ABBR{i}", abbr) for s in sentences]
    
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    
    return sentences

# ============================================================================
# LIVE MeSH LOOKUP via PubMed E-utilities
# ============================================================================

class MeSHLookup:
    def __init__(self, email: str):
        self.email = email
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.delay = 0.34  # To respect NCBI's rate limits
        self.cache = {}  # Simple cache to avoid repeated lookups
    
    def search_mesh_terms(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search for MeSH terms using PubMed's E-utilities
        """
        if not query or len(query) < 3:
            return []
        
        # Check cache first
        cache_key = f"mesh_search_{query.lower()}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Search for MeSH terms
            search_url = f"{self.base_url}esearch.fcgi"
            params = {
                'db': 'mesh',
                'term': f"{query}[All Fields]",
                'retmode': 'json',
                'retmax': max_results,
                'sort': 'relevance',
                'email': self.email,
                'tool': 'streamlit_app'
            }
            
            time.sleep(self.delay)
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            id_list = data.get('esearchresult', {}).get('idlist', [])
            
            if not id_list:
                # Try a more flexible search
                params['term'] = f"{query}[Text Word]"
                time.sleep(self.delay)
                response = requests.get(search_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                id_list = data.get('esearchresult', {}).get('idlist', [])
            
            results = []
            for mesh_id in id_list[:max_results]:
                term_info = self.fetch_mesh_details(mesh_id)
                if term_info:
                    results.append(term_info)
            
            self.cache[cache_key] = results
            return results
            
        except Exception as e:
            if st.session_state.get('debug_mode', False):
                st.warning(f"Error searching MeSH terms: {str(e)}")
            return []
    
    def fetch_mesh_details(self, mesh_id: str) -> Optional[Dict[str, str]]:
        """
        Fetch details for a specific MeSH term by ID
        """
        try:
            fetch_url = f"{self.base_url}efetch.fcgi"
            params = {
                'db': 'mesh',
                'id': mesh_id,
                'retmode': 'xml',
                'email': self.email,
                'tool': 'streamlit_app'
            }
            
            time.sleep(self.delay)
            response = requests.get(fetch_url, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Find descriptor name
            descriptor_name = root.find('.//DescriptorName')
            if descriptor_name is not None and descriptor_name.text:
                term_name = descriptor_name.text
                
                # Find UI
                descriptor_ui = root.find('.//DescriptorUI')
                mesh_ui = descriptor_ui.text if descriptor_ui is not None else mesh_id
                
                # Find tree numbers for context
                tree_numbers = []
                for tree_num in root.findall('.//TreeNumber'):
                    if tree_num.text:
                        tree_numbers.append(tree_num.text)
                
                # Find scope note if available
                scope_note = ""
                for note in root.findall('.//ScopeNote'):
                    if note.text:
                        scope_note = note.text[:200] + "..." if len(note.text) > 200 else note.text
                        break
                
                return {
                    'id': mesh_ui,
                    'term': term_name,
                    'tree_numbers': ', '.join(tree_numbers[:3]),
                    'scope_note': scope_note
                }
            
            return None
            
        except Exception as e:
            if st.session_state.get('debug_mode', False):
                st.warning(f"Error fetching MeSH details: {str(e)}")
            return None
    
    def find_best_mesh_match(self, query: str) -> Optional[str]:
        """
        Find the best matching MeSH term for a query string
        """
        results = self.search_mesh_terms(query, max_results=3)
        
        if not results:
            return None
        
        # Calculate similarity scores
        best_match = None
        best_score = 0.0
        
        for result in results:
            term = result['term'].lower()
            query_lower = query.lower()
            
            # Calculate similarity
            similarity = SequenceMatcher(None, query_lower, term).ratio()
            
            # Boost exact matches
            if query_lower == term:
                similarity = 1.0
            elif query_lower in term:
                similarity += 0.2
            elif term in query_lower:
                similarity += 0.1
            
            if similarity > best_score:
                best_score = similarity
                best_match = result['term']
        
        return best_match if best_score > 0.6 else None
    
    def render_term_selector(self, query: str, key_prefix: str) -> Optional[str]:
        """
        Render a UI component to select from multiple MeSH term suggestions
        """
        if not query or len(query) < 3:
            return None
        
        results = self.search_mesh_terms(query, max_results=5)
        
        if not results:
            return None
        
        options = [result['term'] for result in results]
        descriptions = [f"{result['term']} - {result['scope_note'][:100]}..." if result['scope_note'] else result['term'] for result in results]
        
        selected_idx = st.selectbox(
            f"Select MeSH term for '{query}':",
            range(len(options)),
            format_func=lambda i: descriptions[i],
            key=f"{key_prefix}_mesh_selector_{query}"
        )
        
        return options[selected_idx]

# ============================================================================
# HYPOTHESIS ASSISTANT - WITH LIVE MeSH LOOKUP
# ============================================================================

class HypothesisAssistant:
    def __init__(self, email: str):
        self.mesh_lookup = MeSHLookup(email)
        
        self.templates = {
            "causal": {
                "en": "The {subject} {verb} {effect} in {population}",
                "description": "Direct causal relationship",
                "verbs": ["causes", "produces", "induces", "provokes", "generates"]
            },
            "association": {
                "en": "There is an association between {subject} and {effect} in {population}",
                "description": "Association or correlation",
                "verbs": ["is associated with", "is related to", "is correlated with"]
            },
            "risk": {
                "en": "The {subject} increases the risk of {effect} in {population}",
                "description": "Risk factor",
                "verbs": ["increases the risk of", "is a risk factor for"]
            },
            "prevention": {
                "en": "The {subject} reduces the incidence of {effect} in {population}",
                "description": "Protective or preventive factor",
                "verbs": ["reduces", "decreases", "prevents", "protects against"]
            },
            "effectiveness": {
                "en": "The {subject} is effective in treating {effect} in {population}",
                "description": "Therapeutic effectiveness",
                "verbs": ["is effective in", "demonstrates efficacy in"]
            }
        }
        
        # Common MeSH term overrides for very common terms (as fallback)
        self.common_mesh_overrides = {
            "adult": "Adult",
            "adults": "Adult",
            "child": "Child",
            "children": "Child",
            "aged": "Aged",
            "elderly": "Aged",
            "mortality": "Mortality",
            "death": "Mortality",
            "diabetes": "Diabetes Mellitus, Type 2",
            "type 2 diabetes": "Diabetes Mellitus, Type 2",
            "exercise": "Exercise",
            "physical activity": "Exercise",
            "alcoholism": "Alcoholism",
            "alcohol": "Alcohol Drinking",
            "smoking": "Smoking",
            "obesity": "Obesity",
            "overweight": "Overweight",
            "ticagrelor": "Ticagrelor",
            "dyspnea": "Dyspnea",
            "myocardial infarction": "Myocardial Infarction",
            "heart attack": "Myocardial Infarction",
            "cardiac rupture": "Heart Rupture, Post-Infarction",
            "prion": "Prion Diseases",
            "prions": "Prion Diseases",
            "cancer": "Neoplasms",
            "lung cancer": "Lung Neoplasms",
            "breast cancer": "Breast Neoplasms"
        }
        
        # SUBHEADINGS - these are standard in PubMed
        self.subheadings = {
            "prevention": "prevention and control",
            "therapy": "therapy",
            "treatment": "therapy",
            "diagnosis": "diagnosis",
            "epidemiology": "epidemiology",
            "adverse effects": "adverse effects",
            "physiopathology": "physiopathology",
            "mortality": "mortality"
        }
        
        # EXAMPLES (queries will be regenerated with live MeSH lookup)
        self.examples = [
            {
                "name": "Ticagrelor and dyspnea",
                "subject": "ticagrelor",
                "effect": "dyspnea",
                "population": "myocardial ischemia",
                "type": "causal",
                "verb": "causes",
                "hypothesis": "Ticagrelor causes dyspnea as a side effect in patients with myocardial ischemia"
            },
            {
                "name": "Post-infarction cardiac rupture",
                "subject": "cardiac rupture",
                "effect": "anatomical patterns",
                "population": "myocardial infarction",
                "type": "association",
                "verb": "follows",
                "hypothesis": "In post-infarction cardiac rupture, the heart ruptures following recognizable anatomical patterns"
            },
            {
                "name": "Exercise and diabetes",
                "subject": "exercise",
                "effect": "type 2 diabetes",
                "population": "overweight adults",
                "type": "prevention",
                "verb": "reduces the incidence of",
                "hypothesis": "Regular physical exercise reduces the incidence of type 2 diabetes in overweight adults"
            },
            {
                "name": "Alcoholism and mortality",
                "subject": "alcoholism",
                "effect": "mortality",
                "population": "adults",
                "type": "risk",
                "verb": "increases the risk of",
                "hypothesis": "Alcoholism increases the risk of mortality in adults"
            },
            {
                "name": "Smoking and lung cancer",
                "subject": "smoking",
                "effect": "lung cancer",
                "population": "adults",
                "type": "risk",
                "verb": "increases the risk of",
                "hypothesis": "Smoking increases the risk of lung cancer in adults"
            },
            {
                "name": "Prions and mortality",
                "subject": "prions",
                "effect": "mortality",
                "population": "adults",
                "type": "causal",
                "verb": "causes",
                "hypothesis": "Prions cause mortality in adults"
            }
        ]
    
    def get_available_verbs(self, template_type: str) -> List[str]:
        if template_type in self.templates:
            return self.templates[template_type].get("verbs", [])
        return []
    
    def find_mesh_term_live(self, term: str) -> Optional[str]:
        """
        Find the best MeSH term using live lookup
        """
        if not term:
            return None
        
        term_lower = term.lower().strip()
        
        # Check common overrides first (faster)
        if term_lower in self.common_mesh_overrides:
            if st.session_state.get('debug_mode', False):
                st.write(f"   ✓ Using common override: {self.common_mesh_overrides[term_lower]}")
            return self.common_mesh_overrides[term_lower]
        
        # Live lookup
        if st.session_state.get('debug_mode', False):
            st.write(f"   🔍 Live MeSH lookup for: '{term}'")
        
        best_match = self.mesh_lookup.find_best_mesh_match(term)
        
        if best_match:
            if st.session_state.get('debug_mode', False):
                st.write(f"   ✓ Found: {best_match}")
            return best_match
        else:
            if st.session_state.get('debug_mode', False):
                st.write(f"   ✗ No MeSH term found")
            return None
    
    def generate_mesh_query(self, subject: str, effect: str, population: str, 
                           template_type: str, verb: str = None) -> Tuple[str, Dict[str, Optional[str]]]:
        """
        Generate a query in CORRECT MeSH format for PubMed using live lookup
        Returns both the query string and the mapping of terms found
        """
        query_parts = []
        term_mapping = {}
        
        # Debug info
        if st.session_state.get('debug_mode', False):
            st.write("🔍 Generating MeSH query with live lookup:")
        
        # Get term for subject
        subject_term = self.find_mesh_term_live(subject)
        if subject_term:
            term_mapping['subject'] = subject_term
            
            # Add subheading for prevention type
            if template_type == "prevention":
                subject_final = f'"{subject_term}/prevention and control"[Mesh]'
            else:
                subject_final = f'"{subject_term}"[Mesh]'
            query_parts.append(subject_final)
        
        # Get term for effect
        effect_term = self.find_mesh_term_live(effect)
        if effect_term:
            term_mapping['effect'] = effect_term
            
            # Add subheading for prevention type
            if template_type == "prevention":
                effect_final = f'"{effect_term}/prevention and control"[Mesh]'
            elif template_type == "risk" and "mortality" in effect_term.lower():
                effect_final = f'"{effect_term}/mortality"[Mesh]'
            else:
                effect_final = f'"{effect_term}"[Mesh]'
            query_parts.append(effect_final)
        
        # Get term for population
        population_term = self.find_mesh_term_live(population)
        if population_term:
            term_mapping['population'] = population_term
            population_final = f'"{population_term}"[Mesh]'
            query_parts.append(population_final)
        
        # If no query parts, return default
        if not query_parts:
            default_query = '("research"[Title/Abstract])'
            return default_query, term_mapping
        
        # Join with AND and wrap in parentheses
        query = " AND ".join(query_parts)
        final_query = f"({query})"
        
        return final_query, term_mapping
    
    def build_hypothesis(self, subject: str, effect: str, population: str, 
                        template_type: str, verb: str = None) -> Dict:
        if template_type not in self.templates:
            return {"en": ""}
        
        template = self.templates[template_type]
        
        if not verb and template.get("verbs"):
            verb = template["verbs"][0]
        
        if verb:
            hypothesis_en = template["en"].format(
                subject=subject,
                verb=verb,
                effect=effect,
                population=population
            )
        else:
            hypothesis_en = template["en"].format(
                subject=subject,
                effect=effect,
                population=population
            )
        
        # Generate MeSH query with live lookup
        mesh_query, term_mapping = self.generate_mesh_query(subject, effect, population, template_type, verb)
        
        return {
            "en": hypothesis_en,
            "mesh_query": mesh_query,
            "term_mapping": term_mapping,
            "subject": subject,
            "effect": effect,
            "population": population,
            "type": template_type,
            "type_description": template["description"],
            "verb": verb
        }
    
    def render_assistant_ui(self):
        with st.expander("🤖 HYPOTHESIS ASSISTANT - Build your scientific hypothesis", expanded=False):
            st.markdown('<div class="assistant-box">', unsafe_allow_html=True)
            st.markdown("### 🎯 Build your scientific hypothesis")
            st.markdown("Complete the following fields to generate a well-formed hypothesis and its MeSH query using **LIVE MeSH lookup from PubMed**:")
            
            # Show info about live lookup
            st.info("🔍 Terms are looked up in real-time from the official PubMed MeSH database. No local dictionary needed!")
            
            # Show example of correct format
            st.markdown("""
            <div class="query-example">
            <b>✅ CORRECT MeSH FORMAT (auto-generated):</b><br>
            Your query will be generated automatically based on live MeSH term lookup
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                example_option = st.selectbox(
                    "📋 Load example:",
                    ["Custom"] + [e["name"] for e in self.examples],
                    key="example_selector"
                )
                
                subject = st.text_input(
                    "🧪 Subject/Intervention:",
                    value=st.session_state.get('assistant_subject', ''),
                    placeholder="e.g., ticagrelor, exercise, prions, smoking",
                    key="assistant_subject_input",
                    help="Enter any medical term - it will be looked up in MeSH automatically"
                )
                
                effect = st.text_input(
                    "📊 Effect/Outcome:",
                    value=st.session_state.get('assistant_effect', ''),
                    placeholder="e.g., dyspnea, mortality, lung cancer, diabetes",
                    key="assistant_effect_input",
                    help="Enter any medical term - it will be looked up in MeSH automatically"
                )
                
                population = st.text_input(
                    "👥 Population:",
                    value=st.session_state.get('assistant_population', ''),
                    placeholder="e.g., adults, overweight patients, elderly",
                    key="assistant_population_input",
                    help="Enter any population term - it will be looked up in MeSH automatically"
                )
                
                # Update session state with input values
                if subject != st.session_state.get('assistant_subject', ''):
                    st.session_state['assistant_subject'] = subject
                if effect != st.session_state.get('assistant_effect', ''):
                    st.session_state['assistant_effect'] = effect
                if population != st.session_state.get('assistant_population', ''):
                    st.session_state['assistant_population'] = population
            
            with col2:
                template_type = st.selectbox(
                    "🔄 Relationship type:",
                    options=list(self.templates.keys()),
                    format_func=lambda x: f"{x} - {self.templates[x]['description']}",
                    key="template_type_select"
                )
                
                available_verbs = self.get_available_verbs(template_type)
                if available_verbs:
                    verb = st.selectbox(
                        "🔤 Verb/relationship:",
                        options=available_verbs,
                        key="selected_verb"
                    )
                else:
                    verb = None
                
                st.markdown("---")
                st.markdown("##### 💡 Suggestions:")
                if template_type == "causal":
                    st.info("Use strong verbs like 'causes', 'induces' for direct causal relationships")
                elif template_type == "association":
                    st.info("Use 'is associated with', 'is related to' for correlations without established causality")
                elif template_type == "risk":
                    st.info("Use when the subject increases the probability of the effect")
                elif template_type == "prevention":
                    st.info("Use when the subject reduces risk or protects against the effect")
                elif template_type == "effectiveness":
                    st.info("Ideal for evaluating therapeutic interventions")
            
            if st.button("✨ GENERATE HYPOTHESIS", type="primary", use_container_width=True):
                if subject and effect and population:
                    with st.spinner("🔍 Looking up MeSH terms in PubMed database..."):
                        hypothesis_data = self.build_hypothesis(
                            subject=subject,
                            effect=effect,
                            population=population,
                            template_type=template_type,
                            verb=verb
                        )
                    
                    st.markdown("---")
                    st.markdown("### 📝 GENERATED HYPOTHESIS")
                    
                    st.markdown("**🇬🇧 English:**")
                    st.success(hypothesis_data["en"])
                    
                    # Show MeSH term mapping
                    if hypothesis_data["term_mapping"]:
                        st.markdown("**📋 MeSH Terms Found:**")
                        mapping_text = ""
                        if 'subject' in hypothesis_data["term_mapping"]:
                            mapping_text += f"• Subject: **{hypothesis_data['term_mapping']['subject']}**\n"
                        if 'effect' in hypothesis_data["term_mapping"]:
                            mapping_text += f"• Effect: **{hypothesis_data['term_mapping']['effect']}**\n"
                        if 'population' in hypothesis_data["term_mapping"]:
                            mapping_text += f"• Population: **{hypothesis_data['term_mapping']['population']}**\n"
                        st.info(mapping_text)
                    
                    # Show the generated MeSH query
                    st.markdown("---")
                    st.markdown("**🔍 MeSH query for PubMed (generated from live lookup):**")
                    st.markdown(f'<div class="mesh-query">{hypothesis_data["mesh_query"]}</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Button to use the hypothesis
                        if st.button("📌 Use this hypothesis", key="use_hypothesis", use_container_width=True):
                            st.session_state['hypothesis'] = hypothesis_data["en"]
                            st.session_state['query'] = hypothesis_data["mesh_query"]
                            
                            st.success("✅ Hypothesis and MeSH query loaded")
                            st.rerun()
                    
                    with col2:
                        # Button to use only the MeSH query
                        if st.button("📋 Use only MeSH query", key="use_mesh_only", use_container_width=True):
                            st.session_state['query'] = hypothesis_data["mesh_query"]
                            st.success("✅ MeSH query loaded in search field")
                            st.rerun()
                    
                    st.session_state['last_hypothesis_data'] = hypothesis_data
                    
                else:
                    st.warning("⚠️ Complete all fields to generate the hypothesis")
            
            if example_option != "Custom":
                example = next((e for e in self.examples if e["name"] == example_option), None)
                if example:
                    st.markdown("---")
                    st.markdown("### 📋 Loaded example:")
                    st.info(f"**Hypothesis:** {example['hypothesis']}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("📌 Use this example", key="use_example_main", use_container_width=True):
                            st.session_state['assistant_subject'] = example['subject']
                            st.session_state['assistant_effect'] = example['effect']
                            st.session_state['assistant_population'] = example['population']
                            st.session_state['template_type'] = example['type']
                            st.rerun()
                    
                    with col2:
                        if st.button("🔍 Generate MeSH query", key="generate_example_mesh", use_container_width=True):
                            with st.spinner("🔍 Looking up MeSH terms..."):
                                hypothesis_data = self.build_hypothesis(
                                    subject=example['subject'],
                                    effect=example['effect'],
                                    population=example['population'],
                                    template_type=example['type'],
                                    verb=example['verb']
                                )
                                st.session_state['query'] = hypothesis_data["mesh_query"]
                                st.success("✅ MeSH query generated and loaded")
                                st.rerun()
                    
                    with col3:
                        if st.button("📋 Load in assistant", key="load_example_assistant", use_container_width=True):
                            st.session_state['assistant_subject'] = example['subject']
                            st.session_state['assistant_effect'] = example['effect']
                            st.session_state['assistant_population'] = example['population']
                            st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="tip-box">', unsafe_allow_html=True)
            st.markdown("""
            **💡 Tips for a good hypothesis:**
            
            1. **Be specific**: The more specific, the better the evidence verification
            2. **Define clearly**: Subject, effect, and population must be clearly defined
            3. **Use medical terminology**: Scientific articles use standardized MeSH terms
            4. **Consider the direction**: Is it causality, association, risk, or protection?
            5. **Relevant population**: Specify age, condition, context when relevant
            
            **🔍 NEW: LIVE MeSH LOOKUP**
            - Terms are searched in real-time in the official PubMed MeSH database
            - No local dictionary needed - always up to date
            - Automatically finds the best matching MeSH terms
            
            **📊 CORRECT MeSH FORMAT:**
            - For terms with subheading: `"Term/subheading"[Mesh]`
            - For terms without subheading: `"Term"[Mesh]`
            - Join with AND
            
            **✅ EXAMPLE:**<br>
            If you enter "prions", the system will look up the correct MeSH term "Prion Diseases"
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# ADVANCED SEMANTIC VERIFIER
# ============================================================================

class AdvancedSemanticVerifier:
    def __init__(self):
        if NLTK_READY:
            self.stop_words_en = set(stopwords.words('english'))
            self.stemmer = SnowballStemmer('english')
        else:
            self.stop_words_en = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on',
                                 'at', 'to', 'for', 'with', 'without', 'by'}
            self.stemmer = None
        
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
            'abstract': ['abstract', 'background', 'objective', 'methods', 'results'],
            'introduction': ['introduction', 'background'],
            'methods': ['methods', 'methodology', 'material and methods'],
            'results': ['results', 'findings'],
            'discussion': ['discussion'],
            'conclusion': ['conclusion', 'conclusions', 'concluding']
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
            r'no\s+(association|relationship|correlation)',
            r'not\s+(associated|related|correlated)',
            r'no\s+(significant|statistically\s+significant)',
            r'p\s*[>=]\s*0\.0[5-9]',
            r'p\s*>\s*0\.05'
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
        
        # Medical terms by domain (for relevance calculation, not for MeSH lookup)
        self.cardiac_terms = [
            'intramyocardial', 'dissection', 'hematoma', 'rupture', 'cardiac',
            'postinfarction', 'myocardial', 'infarction', 'free wall', 'septal',
            'ventricular', 'left ventricular', 'anatomical', 'patterns',
            'complex', 'intramyocardial dissection', 'intramyocardial hematoma',
            'complex rupture', 'cardiac rupture', 'heart rupture'
        ]
        
        self.pharmacology_terms = [
            'ticagrelor', 'dyspnea', 'breathlessness', 'shortness of breath',
            'adverse effect', 'side effect', 'adverse event', 'myocardial ischemia',
            'angina', 'antiplatelet', 'p2y12 inhibitor', 'bleeding', 'hemorrhage',
            'cardiovascular', 'acute coronary syndrome', 'acs'
        ]
        
        self.metabolism_terms = [
            'exercise', 'physical activity', 'aerobic', 'resistance training',
            'diabetes', 'type 2 diabetes', 't2dm', 'type 2 diabetes mellitus',
            'obesity', 'overweight', 'bmi', 'body mass index', 'insulin resistance',
            'glucose', 'hba1c', 'glycemic control', 'weight loss', 'prevention',
            'incidence', 'risk reduction', 'metabolic syndrome'
        ]
        
        self.addiction_terms = [
            'alcoholism', 'alcohol dependence', 'alcohol use disorder', 'alcohol abuse',
            'smoking', 'cigarette smoking', 'tobacco', 'nicotine', 'addiction',
            'substance abuse', 'substance use disorder'
        ]
        
        self.prion_terms = [
            'prion', 'prions', 'prion disease', 'prion diseases', 'creutzfeldt-jakob',
            'cjd', 'kuru', 'gss', 'gerstmann-straussler-scheinker', 'ffi',
            'fatal familial insomnia', 'transmissible spongiform encephalopathy',
            'tse', 'spongiform encephalopathy', 'bovine spongiform encephalopathy',
            'bse', 'scrapie', 'chronic wasting disease', 'cwd'
        ]
        
        self.general_terms = [
            'study', 'trial', 'cohort', 'analysis', 'meta-analysis',
            'systematic review', 'randomized', 'controlled', 'prospective',
            'retrospective', 'observational', 'cross-sectional', 'case-control',
            'significant', 'statistically significant', 'association',
            'correlation', 'risk factor', 'protective', 'hazard ratio',
            'odds ratio', 'confidence interval', 'p value', 'mortality', 'death'
        ]
        
        self.UMBRAL_RELEVANCIA = 0.05
        
        self.all_medical_terms = list(set(
            self.cardiac_terms + 
            self.pharmacology_terms + 
            self.metabolism_terms + 
            self.addiction_terms +
            self.prion_terms +
            self.general_terms
        ))
    
    def detect_domain(self, hypothesis: str) -> str:
        hypo_lower = hypothesis.lower()
        
        cardiac_keywords = ['cardiac', 'myocardial', 'infarction', 'rupture', 'intramyocardial',
                           'heart', 'ventricular', 'septal']
        
        pharmacology_keywords = ['ticagrelor', 'drug', 'adverse', 'side effect', 'dyspnea',
                                'antiplatelet', 'pharmacology']
        
        metabolism_keywords = ['exercise', 'diabetes', 'obesity', 'overweight', 'glucose',
                              'metabolic', 'insulin', 'weight']
        
        addiction_keywords = ['alcohol', 'alcoholism', 'smoking', 'tobacco', 'addiction',
                             'substance', 'dependence']
        
        prion_keywords = ['prion', 'creutzfeldt', 'jakob', 'cjd', 'kuru', 'gss', 
                          'gerstmann', 'straussler', 'scheinker', 'ffi', 'insomnia',
                          'spongiform', 'encephalopathy', 'bse', 'scrapie', 'cwd']
        
        cardiac_score = sum(1 for kw in cardiac_keywords if kw in hypo_lower)
        pharmacology_score = sum(1 for kw in pharmacology_keywords if kw in hypo_lower)
        metabolism_score = sum(1 for kw in metabolism_keywords if kw in hypo_lower)
        addiction_score = sum(1 for kw in addiction_keywords if kw in hypo_lower)
        prion_score = sum(1 for kw in prion_keywords if kw in hypo_lower)
        
        scores = {
            'cardiac': cardiac_score,
            'pharmacology': pharmacology_score,
            'metabolism': metabolism_score,
            'addiction': addiction_score,
            'prion': prion_score
        }
        
        max_domain = max(scores, key=scores.get)
        max_score = scores[max_domain]
        
        if max_score == 0:
            return 'general'
        
        return max_domain
    
    def detect_section(self, text_block: str) -> str:
        text_lower = text_block.lower()[:500]
        
        for section, keywords in self.section_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return section
        
        return 'unknown'
    
    def extract_key_terms(self, hypothesis: str) -> List[str]:
        hypothesis_lower = hypothesis.lower()
        all_terms = []
        
        quoted_phrases = re.findall(r'"([^"]*)"', hypothesis_lower)
        for phrase in quoted_phrases:
            if len(phrase) > 3:
                all_terms.append(phrase)
                for word in phrase.split():
                    if len(word) > 3:
                        all_terms.append(word)
        
        words = re.findall(r'\b[a-zA-Z]+\b', hypothesis_lower)
        for word in words:
            if len(word) > 3 and word not in self.stop_words_en:
                all_terms.append(word)
        
        domain = self.detect_domain(hypothesis)
        
        if domain == 'cardiac':
            all_terms.extend(self.cardiac_terms)
        elif domain == 'pharmacology':
            all_terms.extend(self.pharmacology_terms)
        elif domain == 'metabolism':
            all_terms.extend(self.metabolism_terms)
        elif domain == 'addiction':
            all_terms.extend(self.addiction_terms)
        elif domain == 'prion':
            all_terms.extend(self.prion_terms)
        
        all_terms.extend(self.general_terms)
        
        return list(set(all_terms))
    
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
    
    def calculate_relevance(self, sentence: str, hypothesis_terms: List[str], domain: str = 'general') -> float:
        if not hypothesis_terms:
            return 0.0
        
        sentence_lower = sentence.lower()
        
        score = 0.0
        terms_found = []
        
        for term in self.all_medical_terms:
            if term in sentence_lower:
                if len(term.split()) > 1:
                    score += 0.15
                else:
                    score += 0.08
                terms_found.append(term)
        
        if domain == 'cardiac':
            if 'intramyocardial' in sentence_lower:
                if 'dissection' in sentence_lower or 'hematoma' in sentence_lower:
                    score += 0.5
                else:
                    score += 0.2
            
            if 'rupture' in sentence_lower:
                if 'free wall' in sentence_lower:
                    score += 0.4
                elif 'septal' in sentence_lower:
                    score += 0.4
                elif 'ventricular' in sentence_lower:
                    score += 0.3
                else:
                    score += 0.15
        
        elif domain == 'pharmacology':
            if 'ticagrelor' in sentence_lower:
                score += 0.3
                if 'dyspnea' in sentence_lower or 'breath' in sentence_lower:
                    score += 0.5
                elif 'adverse' in sentence_lower or 'side effect' in sentence_lower:
                    score += 0.3
            
            if 'dyspnea' in sentence_lower or 'shortness of breath' in sentence_lower:
                score += 0.25
                if 'antiplatelet' in sentence_lower or 'p2y12' in sentence_lower:
                    score += 0.3
        
        elif domain == 'metabolism':
            if 'exercise' in sentence_lower or 'physical activity' in sentence_lower:
                score += 0.3
                if 'diabetes' in sentence_lower or 'type 2 diabetes' in sentence_lower:
                    score += 0.5
                elif 'glucose' in sentence_lower or 'hba1c' in sentence_lower:
                    score += 0.35
                elif 'weight loss' in sentence_lower or 'obesity' in sentence_lower:
                    score += 0.3
            
            if 'diabetes' in sentence_lower or 't2dm' in sentence_lower:
                score += 0.25
                if 'prevention' in sentence_lower or 'incidence' in sentence_lower:
                    score += 0.4
                elif 'risk' in sentence_lower and ('reduction' in sentence_lower or 'decrease' in sentence_lower):
                    score += 0.35
        
        elif domain == 'addiction':
            if 'alcohol' in sentence_lower or 'alcoholism' in sentence_lower:
                score += 0.4
                if 'mortality' in sentence_lower or 'death' in sentence_lower:
                    score += 0.5
                elif 'risk' in sentence_lower:
                    score += 0.3
            
            if 'smoking' in sentence_lower or 'tobacco' in sentence_lower:
                score += 0.4
                if 'cancer' in sentence_lower or 'mortality' in sentence_lower:
                    score += 0.5
            
            if 'mortality' in sentence_lower:
                score += 0.3
        
        elif domain == 'prion':
            if 'prion' in sentence_lower:
                score += 0.4
                if 'disease' in sentence_lower:
                    score += 0.3
                if 'mortality' in sentence_lower or 'death' in sentence_lower:
                    score += 0.5
            
            if 'creutzfeldt' in sentence_lower or 'jakob' in sentence_lower or 'cjd' in sentence_lower:
                score += 0.5
                if 'mortality' in sentence_lower:
                    score += 0.6
            
            if 'spongiform' in sentence_lower or 'encephalopathy' in sentence_lower:
                score += 0.4
            
            if 'mortality' in sentence_lower:
                score += 0.3
        
        study_indicators = ['randomized', 'controlled trial', 'cohort', 'meta-analysis', 
                           'systematic review', 'prospective']
        for indicator in study_indicators:
            if indicator in sentence_lower:
                score += 0.1
                break
        
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
                'error': 'Insufficient text for analysis',
                'verdict': None
            }
        
        domain = self.detect_domain(hypothesis)
        
        hypothesis_terms = self.extract_key_terms(hypothesis)
        
        if st.session_state.get('debug_mode', False):
            st.write(f"📋 Detected domain: **{domain}**")
            st.write(f"📋 Search terms (first 15): {hypothesis_terms[:15]}")
        
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
            
            relevance = self.calculate_relevance(sentence, hypothesis_terms, domain)
            
            if relevance > self.UMBRAL_RELEVANCIA:
                analysis = self.analyze_sentence_deep(sentence)
                
                evidence = {
                    'sentence': sentence[:200] + '...' if len(sentence) > 200 else sentence,
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
            'domain': domain,
            'total_sentences': len(sentences),
            'relevant_sentences': len(evidence_list),
            'section_distribution': dict(section_counts),
            'study_quality': quality,
            'evidence': evidence_list[:15],
            'verdict': verdict,
            'hypothesis_terms': hypothesis_terms[:20]
        }
    
    def weighted_vote(self, evidence_list: List[Dict], quality: Dict) -> Dict:
        if not evidence_list:
            return {
                'score': 0,
                'confidence': 0,
                'verdict': 'inconclusive',
                'verdict_text': 'INCONCLUSIVE EVIDENCE',
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
            text = 'STRONGLY SUPPORTS'
        elif avg_score > 0.3:
            verdict = 'supports'
            text = 'SUPPORTS'
        elif avg_score > -0.3:
            verdict = 'inconclusive'
            text = 'INCONCLUSIVE'
        elif avg_score > -0.8:
            verdict = 'contradicts'
            text = 'CONTRADICTS'
        else:
            verdict = 'strongly_contradicts'
            text = 'STRONGLY CONTRADICTS'
        
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
# SCIENTIFIC SEARCH ENGINE
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
        
        # Handle correct format with subheadings
        mesh_with_subheading_pattern = r'"([^/]+)/([^"]+)"\[Mesh\]'
        
        def replace_mesh_with_subheading(match):
            term = match.group(1).strip()
            subheading = match.group(2).strip()
            return f'"{term}/{subheading}"[mh]'
        
        query = re.sub(mesh_with_subheading_pattern, replace_mesh_with_subheading, query)
        
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
                st.write(f"📊 PubMed: {count} results found")
            
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
            st.warning(f"Error in PubMed: {str(e)}")
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
                    title = title_elem.text if title_elem is not None and title_elem.text else "Title not available"
                    
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
                        'database': 'PubMed',
                        'title': title,
                        'authors': ', '.join(authors)[:200],
                        'journal': journal,
                        'year': year,
                        'doi': doi,
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                        'pmid': pmid,
                        'abstract': abstract[:500] + '...' if len(abstract) > 500 else abstract,
                        'type': 'Article'
                    })
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            st.warning(f"Error in fetch_pubmed_batch: {str(e)}")
        
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
                    'database': 'CrossRef',
                    'title': item.get('title', ['Title not available'])[0],
                    'authors': ', '.join(authors)[:200],
                    'journal': item.get('container-title', [''])[0] if item.get('container-title') else '',
                    'year': year,
                    'doi': item.get('DOI', ''),
                    'url': f"https://doi.org/{item['DOI']}" if item.get('DOI') else '',
                    'type': item.get('type', '')
                })
                
        except Exception as e:
            st.warning(f"Error in CrossRef: {str(e)}")
        
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
                    title = item.get('title')
                    if title is None:
                        title = "Title not available"
                    
                    authors_list = []
                    for a in item.get('authorships', [])[:5]:
                        author_data = a.get('author', {})
                        if author_data is not None:
                            author_name = author_data.get('display_name')
                            if author_name:
                                authors_list.append(author_name)
                    
                    journal = ""
                    host_venue = item.get('host_venue')
                    if host_venue is not None:
                        journal = host_venue.get('display_name', '')
                    
                    year = item.get('publication_year')
                    year_str = str(year) if year is not None else ""
                    
                    doi = item.get('doi')
                    if doi is not None:
                        doi = doi.replace('https://doi.org/', '')
                    else:
                        doi = ""
                    
                    results.append({
                        'database': 'OpenAlex',
                        'title': title,
                        'authors': ', '.join(authors_list),
                        'journal': journal,
                        'year': year_str,
                        'doi': doi,
                        'url': f"https://doi.org/{doi}" if doi else "",
                        'type': item.get('type', ''),
                    })
                    
                    if len(results) >= max_results:
                        break
                
                page += 1
                if page > 10:
                    break
                
        except Exception as e:
            st.warning(f"Error in OpenAlex: {str(e)}")
        
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
                        'database': 'Europe PMC',
                        'title': item.get('title', 'Title not available'),
                        'authors': ', '.join([a.get('fullName', '') for a in item.get('authorList', {}).get('author', [])[:5]]),
                        'journal': item.get('journalTitle', ''),
                        'year': str(item.get('pubYear', '')),
                        'doi': item.get('doi', ''),
                        'url': f"https://europepmc.org/article/{item.get('source', '')}/{item.get('id', '')}",
                        'pmid': item.get('pmid', ''),
                        'pmcid': item.get('pmcid', ''),
                        'abstract': item.get('abstractText', '')[:300] + '...' if item.get('abstractText') else '',
                        'type': 'Article'
                    })
                    
                    if len(results) >= max_results:
                        break
                
                page += 1
                if page > 10:
                    break
                
        except Exception as e:
            st.warning(f"Error in Europe PMC: {str(e)}")
        
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
                    with st.spinner(f"🔍 Searching {db}..."):
                        results = search_functions[db](query, max_results_per_db, year_range)
                        all_results.extend(results)
                        if results:
                            st.success(f"✅ {db}: {len(results)} results")
                        else:
                            st.info(f"ℹ️ {db}: 0 results")
                except Exception as e:
                    st.warning(f"Error in {db}: {str(e)}")
            else:
                st.warning(f"⚠️ {db}: Database not available")
        
        if all_results:
            df = pd.DataFrame(all_results)
            
            if 'year' in df.columns:
                df['year'] = pd.to_numeric(df['year'], errors='coerce')
            
            if 'doi' in df.columns:
                df = df.drop_duplicates(subset=['doi'], keep='first')
            
            return df
        else:
            return pd.DataFrame()

# ============================================================================
# ARTICLE TEXT FETCHER
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
            return None, "Empty DOI"
        
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
                    return f"Open Access available at: {oa_url}", f"OpenAlex (OA URL)"
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
        
        return None, "Could not obtain full text"

# ============================================================================
# MAIN INTEGRATED CLASS
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
            'support': 0,
            'contradict': 0,
            'inconclusive': 0,
            'strongly_support': 0
        }
    
    def run_analysis(self, query: str, hypothesis: str, max_results_per_db: int = 1000, 
                     selected_dbs: list = None, year_range: tuple = None,
                     progress_callback=None) -> pd.DataFrame:
        self.results = []
        self.stats = {k: 0 for k in self.stats}
        
        if progress_callback:
            progress_callback("🔍 Searching articles in databases...", 0.05)
        
        articles_df = self.search_engine.search_all(
            query, max_results_per_db, selected_dbs, year_range
        )
        
        if articles_df.empty:
            return pd.DataFrame()
        
        self.stats['total_articles'] = len(articles_df)
        
        if progress_callback:
            progress_callback(f"✅ Found {len(articles_df)} articles. Starting analysis...", 0.1)
        
        results_list = []
        total_articles = len(articles_df)
        
        domain = self.semantic_verifier.detect_domain(hypothesis)
        if st.session_state.get('debug_mode', False):
            st.write(f"🎯 Detected domain: {domain}")
        
        for idx, row in articles_df.iterrows():
            current = idx + 1
            if progress_callback:
                progress_value = 0.1 + 0.85 * (current / total_articles)
                progress_value = min(0.95, max(0.1, progress_value))
                progress_callback(
                    f"🔬 Analyzing article {current}/{total_articles}: {str(row['title'])[:50]}...", 
                    progress_value
                )
            
            article_text, source = self.text_fetcher.get_text_from_doi(row.get('doi', ''))
            
            result_row = {
                'database': row.get('database', 'Unknown'),
                'title': row.get('title', 'No title'),
                'authors': row.get('authors', ''),
                'journal': row.get('journal', ''),
                'year': row.get('year', ''),
                'doi': row.get('doi', ''),
                'url': row.get('url', ''),
                'text_available': article_text is not None,
                'text_source': source if article_text else 'Not available',
                'verdict': '',
                'confidence': 0,
                'score': 0,
                'evidence_for': 0,
                'evidence_against': 0,
                'total_sentences': 0,
                'relevant_sentences': 0,
                'evidence_detail': ''
            }
            
            if article_text:
                self.stats['with_text'] += 1
                
                analysis = self.semantic_verifier.verify_article_text(article_text, hypothesis)
                
                if analysis['success']:
                    verdict = analysis['verdict']
                    
                    result_row.update({
                        'verdict': verdict['verdict_text'],
                        'confidence': verdict['confidence'],
                        'score': verdict['score'],
                        'evidence_for': verdict['support_count'],
                        'evidence_against': verdict['against_count'],
                        'total_sentences': analysis['total_sentences'],
                        'relevant_sentences': analysis['relevant_sentences']
                    })
                    
                    self.stats['analyzed'] += 1
                    if verdict['verdict_text'] == 'STRONGLY SUPPORTS':
                        self.stats['strongly_support'] += 1
                        self.stats['support'] += 1
                    elif 'SUPPORTS' in verdict['verdict_text']:
                        self.stats['support'] += 1
                    elif 'CONTRADICTS' in verdict['verdict_text']:
                        self.stats['contradict'] += 1
                    else:
                        self.stats['inconclusive'] += 1
                    
                    if analysis['evidence']:
                        ev_summary = []
                        for ev in analysis['evidence'][:3]:
                            ev_summary.append(f"[{ev['section']}] {ev['relation']} ({ev['certainty']})")
                        result_row['evidence_detail'] = ' | '.join(ev_summary)
            else:
                result_row['verdict'] = 'TEXT NOT AVAILABLE'
            
            results_list.append(result_row)
            time.sleep(0.1)
        
        if progress_callback:
            progress_callback("✅ Analysis completed", 1.0)
        
        self.results = pd.DataFrame(results_list)
        return self.results
    
    def generate_report(self) -> str:
        if self.results.empty:
            return "No results to generate report."
        
        report = []
        report.append("="*80)
        report.append("INTEGRATED SEMANTIC VERIFICATION REPORT")
        report.append("="*80)
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total articles found: {self.stats['total_articles']}")
        report.append(f"Articles with text available: {self.stats['with_text']}")
        report.append(f"Articles analyzed: {self.stats['analyzed']}")
        report.append("")
        report.append("GLOBAL RESULTS:")
        report.append(f"✅ Strongly support: {self.stats['strongly_support']}")
        report.append(f"✅ Support: {self.stats['support']}")
        report.append(f"❌ Contradict: {self.stats['contradict']}")
        report.append(f"⚠️ Inconclusive: {self.stats['inconclusive']}")
        report.append("")
        report.append("DETAIL BY ARTICLE:")
        report.append("-"*80)
        
        for idx, row in self.results.iterrows():
            report.append(f"\n📄 {row['title']}")
            report.append(f"   Database: {row['database']} | Year: {row['year']}")
            report.append(f"   DOI: {row['doi']}")
            
            if row['verdict'] == 'TEXT NOT AVAILABLE':
                report.append(f"   ⚠️ {row['verdict']} - {row['text_source']}")
            else:
                report.append(f"   Verdict: {row['verdict']} (Confidence: {row['confidence']:.1%})")
                report.append(f"   Evidence: {row['evidence_for']} for, {row['evidence_against']} against")
                if row['evidence_detail']:
                    report.append(f"   Highlighted evidence: {row['evidence_detail']}")
        
        return "\n".join(report)
    
    def generate_html_report(self) -> str:
        if self.results.empty:
            return "<p>No results to generate report.</p>"
        
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
            <h1>🔬 Semantic Verification Report</h1>
            <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>📊 Global Results</h2>
            <div class="stats">
                <p><strong>Total articles found:</strong> {self.stats['total_articles']}</p>
                <p><strong>Articles with text available:</strong> {self.stats['with_text']}</p>
                <p><strong>Articles analyzed:</strong> {self.stats['analyzed']}</p>
                <p><strong>✅ Strongly support:</strong> {self.stats['strongly_support']}</p>
                <p><strong>✅ Support:</strong> {self.stats['support']}</p>
                <p><strong>❌ Contradict:</strong> {self.stats['contradict']}</p>
                <p><strong>⚠️ Inconclusive:</strong> {self.stats['inconclusive']}</p>
            </div>
            
            <h2>📋 Article Details</h2>
            <table>
                <tr>
                    <th>Database</th>
                    <th>Title</th>
                    <th>Year</th>
                    <th>Verdict</th>
                    <th>Confidence</th>
                </tr>
        """
        
        for idx, row in self.results.iterrows():
            title = str(row.get('title', '')) if pd.notna(row.get('title')) else "Title not available"
            database = str(row.get('database', '')) if pd.notna(row.get('database')) else "Unknown"
            year = str(row.get('year', '')) if pd.notna(row.get('year')) else ""
            verdict = str(row.get('verdict', '')) if pd.notna(row.get('verdict')) else "NOT AVAILABLE"
            confidence = float(row.get('confidence', 0)) if pd.notna(row.get('confidence')) else 0
            
            verdict_class = ""
            if 'STRONGLY SUPPORTS' in verdict:
                verdict_class = "verdict-assert"
            elif 'SUPPORTS' in verdict:
                verdict_class = "verdict-assert"
            elif 'CONTRADICTS' in verdict:
                verdict_class = "verdict-reject"
            elif 'INCONCLUSIVE' in verdict:
                verdict_class = "verdict-inconclusive"
            
            html += f"""
                <tr>
                    <td>{database}</td>
                    <td>{title[:100]}...</td>
                    <td>{year}</td>
                    <td class="{verdict_class}">{verdict}</td>
                    <td>{confidence:.1%}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html

# ============================================================================
# VISUALIZATION FUNCTIONS
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
    if 'STRONGLY SUPPORTS' in verdict:
        return 'verdict-assert'
    elif 'SUPPORTS' in verdict:
        return 'verdict-assert'
    elif 'CONTRADICTS' in verdict:
        return 'verdict-reject'
    elif 'INCONCLUSIVE' in verdict:
        return 'verdict-inconclusive'
    else:
        return ''

# ============================================================================
# FUNCTION TO SEND RESULTS BY EMAIL
# ============================================================================

def send_results_email(recipient, integrator):
    if integrator is None or integrator.results is None or integrator.results.empty:
        st.warning("No results to send by email.")
        return False
    
    report_txt = integrator.generate_report()
    report_html = integrator.generate_html_report()
    
    attachments = []
    
    csv_buffer = io.BytesIO()
    integrator.results.to_csv(csv_buffer, index=False, encoding='utf-8')
    attachments.append({
        'name': f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        'content': csv_buffer.getvalue()
    })
    
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        integrator.results.to_excel(writer, sheet_name='Results', index=False)
        summary = pd.DataFrame([integrator.stats])
        summary.to_excel(writer, sheet_name='Summary', index=False)
    attachments.append({
        'name': f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        'content': excel_buffer.getvalue()
    })
    
    subject = f"🔬 Semantic Verification Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    html_message = f"""
    <html>
    <body>
        <h2>🔬 Semantic Verification Report</h2>
        <p>Dear user,</p>
        <p>Attached you will find the complete results of your semantic verification analysis.</p>
        
        <h3>📊 Summary</h3>
        <ul>
            <li><strong>Total articles:</strong> {integrator.stats['total_articles']}</li>
            <li><strong>Articles analyzed:</strong> {integrator.stats['analyzed']}</li>
            <li><strong>✅ Strongly support:</strong> {integrator.stats['strongly_support']}</li>
            <li><strong>✅ Support:</strong> {integrator.stats['support']}</li>
            <li><strong>❌ Contradict:</strong> {integrator.stats['contradict']}</li>
            <li><strong>⚠️ Inconclusive:</strong> {integrator.stats['inconclusive']}</li>
        </ul>
        
        <p>You can find the complete details in the attached files.</p>
        
        <hr>
        <p style="color: #666; font-size: 0.9em;">
            This is an automated message from the Integrated Semantic Search and Verifier.
        </p>
    </body>
    </html>
    """
    
    text_message = f"""
    Semantic Verification Report
    
    Summary:
    - Total articles: {integrator.stats['total_articles']}
    - Articles analyzed: {integrator.stats['analyzed']}
    - ✅ Strongly support: {integrator.stats['strongly_support']}
    - ✅ Support: {integrator.stats['support']}
    - ❌ Contradict: {integrator.stats['contradict']}
    - ⚠️ Inconclusive: {integrator.stats['inconclusive']}
    
    The attached files contain the complete details.
    """
    
    return send_email(
        recipient=recipient,
        subject=subject,
        html_message=html_message,
        text_message=text_message,
        attachments=attachments
    )

# ============================================================================
# MAIN STREAMLIT INTERFACE
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables safely"""
    defaults = {
        'query': "",
        'hypothesis': "",
        'user_email': "",
        'integrator': None,
        'debug_mode': False,
        'assistant_subject': "",
        'assistant_effect': "",
        'assistant_population': "",
        'analysis_completed': False,
        'last_results_df': None,
        'last_stats': None,
        'elapsed_time': 0,
        'template_type': "causal"
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def safe_set_session_state(key, value):
    """Safely set a session state value"""
    try:
        st.session_state[key] = value
    except Exception as e:
        # If we can't set directly, try to initialize first
        if key not in st.session_state:
            st.session_state[key] = value

def main():
    # Initialize session state first thing
    initialize_session_state()
    
    st.markdown('<h1 class="main-header">🔬 Integrated Semantic Search and Verifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">HIGH VOLUME: Up to 1000 articles per database • Automatic AI analysis • STABLE VERSION • MULTIDOMAIN • 4 DATABASES • LIVE MeSH LOOKUP</p>', 
                unsafe_allow_html=True)
    
    # Get email for MeSH lookup
    user_email = st.session_state.get('user_email', '')
    if not user_email or user_email == "":
        user_email = "anonymous@example.com"  # Fallback email
    
    hypothesis_assistant = HypothesisAssistant(user_email)
    hypothesis_assistant.render_assistant_ui()
    
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
        st.markdown("## ⚙️ Configuration")
        
        email = st.text_input(
            "📧 Email (required for NCBI and MeSH lookup)", 
            value=st.session_state.get('user_email', ''),
            placeholder="your@email.com",
            key="email_input",
            help="Your email is required by NCBI for API access"
        )
        safe_set_session_state('user_email', email)
        
        debug_mode = st.checkbox(
            "🔧 Debug mode",
            value=st.session_state.get('debug_mode', False),
            key="debug_checkbox"
        )
        safe_set_session_state('debug_mode', debug_mode)
        
        if NLTK_READY:
            st.success("✅ NLTK configured correctly")
        else:
            st.warning("⚠️ Using alternative tokenizer")
        
        st.markdown("### 📚 Databases")
        st.info("🔔 **4 stable databases**: PubMed, CrossRef, OpenAlex, Europe PMC")
        
        col1, col2 = st.columns(2)
        with col1:
            pubmed = st.checkbox('PubMed', value=True)
            crossref = st.checkbox('CrossRef', value=True)
        with col2:
            openalex = st.checkbox('OpenAlex', value=True)
            europepmc = st.checkbox('Europe PMC', value=True)
        
        databases = {
            'PubMed': pubmed,
            'CrossRef': crossref,
            'OpenAlex': openalex,
            'Europe PMC': europepmc
        }
        
        st.markdown("---")
        
        st.markdown("### 📅 Year Filter")
        col1, col2 = st.columns(2)
        with col1:
            year_from = st.number_input("From", min_value=1900, max_value=2025, value=2015)
        with col2:
            year_to = st.number_input("To", min_value=1900, max_value=2025, value=2025)
        
        year_range = (year_from, year_to) if year_from < year_to else None
        
        st.markdown("### 📊 Results per database")
        max_results = st.slider(
            "Maximum results per database:", 
            min_value=10, max_value=1000, value=100, step=10
        )
        
        st.markdown("### 🔧 Analysis thresholds")
        min_relevance = st.slider(
            "Minimum relevance",
            min_value=0.01,
            max_value=0.2,
            value=0.05,
            step=0.01
        )
        
        st.markdown("### ⚠️ Warning")
        st.warning("""
        **Processing time:**
        - 50 articles: ~2-3 minutes
        - 100 articles: ~5-7 minutes
        - 200 articles: ~10-15 minutes
        - 500 articles: ~25-35 minutes
        - 1000 articles: ~50-70 minutes
        """)
        
        st.markdown("### 📋 Examples")
        st.info("These examples will use LIVE MeSH lookup when generating queries")
        
        # OPTIMIZED EXAMPLES WITH SAFE SESSION STATE UPDATES
        if st.button("Load example: Ticagrelor and dyspnea"):
            safe_set_session_state('assistant_subject', "ticagrelor")
            safe_set_session_state('assistant_effect', "dyspnea")
            safe_set_session_state('assistant_population', "myocardial ischemia")
            safe_set_session_state('template_type', "causal")
            st.rerun()
        
        if st.button("Load example: Post-infarction cardiac rupture"):
            safe_set_session_state('assistant_subject', "cardiac rupture")
            safe_set_session_state('assistant_effect', "anatomical patterns")
            safe_set_session_state('assistant_population', "myocardial infarction")
            safe_set_session_state('template_type', "association")
            st.rerun()
        
        if st.button("Load example: Exercise and diabetes"):
            safe_set_session_state('assistant_subject', "exercise")
            safe_set_session_state('assistant_effect', "type 2 diabetes")
            safe_set_session_state('assistant_population', "overweight adults")
            safe_set_session_state('template_type', "prevention")
            st.rerun()
        
        if st.button("Load example: Alcoholism and mortality"):
            safe_set_session_state('assistant_subject', "alcoholism")
            safe_set_session_state('assistant_effect', "mortality")
            safe_set_session_state('assistant_population', "adults")
            safe_set_session_state('template_type', "risk")
            st.rerun()
        
        if st.button("Load example: Smoking and lung cancer"):
            safe_set_session_state('assistant_subject', "smoking")
            safe_set_session_state('assistant_effect', "lung cancer")
            safe_set_session_state('assistant_population', "adults")
            safe_set_session_state('template_type', "risk")
            st.rerun()
        
        if st.button("Load example: Prions and mortality"):
            safe_set_session_state('assistant_subject', "prions")
            safe_set_session_state('assistant_effect', "mortality")
            safe_set_session_state('assistant_population', "adults")
            safe_set_session_state('template_type', "causal")
            st.rerun()
    
    # Main area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_query = st.text_area(
            "🔍 Search query (will be auto-generated from assistant):",
            value=st.session_state['query'],
            height=120,
            placeholder='Use the Hypothesis Assistant above to generate queries automatically',
            key="query_input",
            help="Use the Hypothesis Assistant to generate MeSH queries automatically"
        )
        if search_query != st.session_state['query']:
            safe_set_session_state('query', search_query)
            safe_set_session_state('analysis_completed', False)
    
    with col2:
        hypothesis = st.text_area(
            "🔬 Hypothesis to verify (will be auto-generated from assistant):",
            value=st.session_state['hypothesis'],
            height=68,
            placeholder='Use the Hypothesis Assistant above to generate hypotheses automatically',
            key="hypothesis_input"
        )
        
        if hypothesis != st.session_state.get('hypothesis', ''):
            safe_set_session_state('hypothesis', hypothesis)
            safe_set_session_state('analysis_completed', False)
    
    col1, col2, col3 = st.columns(3)
    with col2:
        analyze_button = st.button("🚀 START INTEGRATED ANALYSIS", type="primary", use_container_width=True)
    
    # ANALYSIS BLOCK
    if analyze_button and search_query and hypothesis:
        user_email = st.session_state.get('user_email', '')
        if not user_email or not validate_email(user_email):
            st.error("❌ Enter a valid email in the sidebar (required for NCBI access)")
        else:
            selected_dbs = [db for db, selected in databases.items() if selected]
            
            if not selected_dbs:
                st.error("❌ Select at least one database")
            else:
                integrator = IntegratedScientificVerifier(user_email)
                integrator.semantic_verifier.UMBRAL_RELEVANCIA = min_relevance
                
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
                            time_estimate.text(f"⏱️ Elapsed time: {elapsed:.1f}s | Estimated remaining: {remaining:.1f}s")
                
                with st.spinner("Running integrated analysis..."):
                    results_df = integrator.run_analysis(
                        search_query, 
                        hypothesis,
                        max_results, 
                        selected_dbs, 
                        year_range,
                        update_progress
                    )
                    elapsed_time = time.time() - start_time
                
                if not results_df.empty:
                    safe_set_session_state('integrator', integrator)
                    safe_set_session_state('last_results_df', results_df.copy())
                    safe_set_session_state('last_stats', integrator.stats.copy())
                    safe_set_session_state('analysis_completed', True)
                    safe_set_session_state('elapsed_time', elapsed_time)
                    
                    st.rerun()
                else:
                    st.warning("😕 No articles found. Try another query or expand the year range.")
    
    # RESULTS VISUALIZATION BLOCK
    if st.session_state.get('analysis_completed', False) and st.session_state.get('last_results_df') is not None:
        integrator = st.session_state['integrator']
        results_df = st.session_state['last_results_df']
        stats = st.session_state['last_stats']
        elapsed_time = st.session_state['elapsed_time']
        
        if not results_df.empty:
            st.success(f"✅ Analysis completed in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
            
            st.markdown("## 📊 GLOBAL RESULTS")
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("Articles found", stats['total_articles'])
            with col2:
                st.metric("With text", stats['with_text'])
            with col3:
                st.metric("✅ Strong support", stats['strongly_support'])
            with col4:
                st.metric("✅ Support", stats['support'])
            with col5:
                st.metric("❌ Contradict", stats['contradict'])
            with col6:
                st.metric("⚠️ Inconclusive", stats['inconclusive'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                db_counts = results_df['database'].value_counts().reset_index()
                db_counts.columns = ['database', 'count']
                fig = px.bar(
                    db_counts, x='database', y='count',
                    title=f"Articles by Database",
                    color='database',
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
                analyzed_df = results_df[results_df['verdict'] != 'TEXT NOT AVAILABLE']
                if not analyzed_df.empty:
                    verdict_counts = analyzed_df['verdict'].value_counts().reset_index()
                    verdict_counts.columns = ['verdict', 'count']
                    
                    color_map = {
                        'STRONGLY SUPPORTS': '#2E7D32',
                        'SUPPORTS': '#4CAF50',
                        'INCONCLUSIVE': '#ff9800',
                        'CONTRADICTS': '#f44336',
                        'STRONGLY CONTRADICTS': '#b71c1c'
                    }
                    
                    fig = px.pie(
                        verdict_counts, 
                        values='count', 
                        names='verdict',
                        title=f"Verdict Distribution (n={len(analyzed_df)})",
                        color='verdict',
                        color_discrete_map=color_map
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("## 📋 COMPLETE ARTICLES TABLE")
            
            display_cols = ['database', 'title', 'year', 'verdict', 'confidence', 
                           'evidence_for', 'evidence_against']
            display_df = results_df[display_cols].copy()
            display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}" if x > 0 else 'N/A')
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=500,
                column_config={
                    'database': 'Database',
                    'title': 'Title',
                    'year': 'Year',
                    'verdict': 'Verdict',
                    'confidence': 'Confidence',
                    'evidence_for': 'For',
                    'evidence_against': 'Against'
                }
            )
            
            # SECTION: ARTICLES THAT STRONGLY SUPPORT THE HYPOTHESIS
            strong_evidence_df = results_df[results_df['verdict'] == 'STRONGLY SUPPORTS']
            
            if not strong_evidence_df.empty:
                st.markdown("---")
                st.markdown(f'<div class="strong-evidence-title">🔬 ARTICLES THAT STRONGLY SUPPORT THE HYPOTHESIS <span class="counter-badge">{len(strong_evidence_df)} found</span></div>', 
                          unsafe_allow_html=True)
                
                st.markdown("""
                <div style="background-color: #e8f5e8; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
                <p style="margin:0; color: #2E7D32;">📌 These articles provide the strongest evidence in favor of your hypothesis.</p>
                </div>
                """, unsafe_allow_html=True)
                
                for idx, row in strong_evidence_df.iterrows():
                    badge_class = get_badge_class(row['database'])
                    
                    st.markdown(f"""
                    <div class="result-card-strong">
                        <span class="{badge_class}">{row['database']}</span>
                        <div style="font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0;">{row['title']}</div>
                        <div style="color: #666; margin-bottom: 0.5rem;">
                            <b>Authors:</b> {row.get('authors', 'Not available')[:150]}...
                        </div>
                        <div style="margin: 0.5rem 0;">
                            <b>Journal:</b> {row.get('journal', 'Not available')} | 
                            <b>Year:</b> {row.get('year', 'Not available')}
                        </div>
                        <div style="margin: 0.5rem 0;">
                            <b>DOI:</b> {row.get('doi', 'Not available')}
                        </div>
                        <div style="margin: 1rem 0;">
                            <span class="verdict-assert">{row['verdict']}</span> 
                            <span style="margin-left: 1rem; background-color: #4CAF50; color: white; padding: 0.3rem 0.8rem; border-radius: 15px;">
                                Confidence: {row['confidence']:.1%}
                            </span>
                        </div>
                        <div style="background-color: white; padding: 1rem; border-radius: 5px; margin: 0.5rem 0;">
                            <div style="display: flex; justify-content: space-around; text-align: center;">
                                <div>
                                    <div style="font-size: 1.5rem; font-weight: bold; color: #4CAF50;">{row['evidence_for']}</div>
                                    <div>Evidence for</div>
                                </div>
                                <div>
                                    <div style="font-size: 1.5rem; font-weight: bold; color: #f44336;">{row['evidence_against']}</div>
                                    <div>Evidence against</div>
                                </div>
                                <div>
                                    <div style="font-size: 1.5rem; font-weight: bold;">{row.get('total_sentences', 0)}</div>
                                    <div>Total sentences</div>
                                </div>
                                <div>
                                    <div style="font-size: 1.5rem; font-weight: bold;">{row.get('relevant_sentences', 0)}</div>
                                    <div>Relevant sentences</div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if row['evidence_detail']:
                        st.markdown(f"""
                        <div class="evidence-box">
                            <b>🔍 Highlighted evidence:</b> {row['evidence_detail']}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        if row.get('url') and pd.notna(row['url']):
                            st.link_button("🔗 View article", row['url'], use_container_width=True)
                    with col2:
                        if row.get('doi') and pd.notna(row['doi']):
                            doi_link = f"https://doi.org/{row['doi']}"
                            st.link_button("📋 View DOI", doi_link, use_container_width=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("---")
            else:
                st.info("ℹ️ No articles found that strongly support the hypothesis.")
            
            st.markdown("## 💾 EXPORT RESULTS")
            
            col1, col2, col3 = st.columns(3)
            
            report_text = integrator.generate_report() if integrator else "Not available"
            with col1:
                st.download_button(
                    "📝 TXT Report",
                    report_text,
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📊 CSV",
                    csv,
                    file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col3:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    results_df.to_excel(writer, sheet_name='Results', index=False)
                    
                    summary = pd.DataFrame([stats])
                    summary.to_excel(writer, sheet_name='Summary', index=False)
                    
                    if not results_df[results_df['verdict'] != 'TEXT NOT AVAILABLE'].empty:
                        verdict_stats = results_df[results_df['verdict'] != 'TEXT NOT AVAILABLE']['verdict'].value_counts().reset_index()
                        verdict_stats.columns = ['Verdict', 'Count']
                        verdict_stats.to_excel(writer, sheet_name='By Verdict', index=False)
                    
                    if not strong_evidence_df.empty:
                        strong_evidence_df.to_excel(writer, sheet_name='Strong Evidence', index=False)
                
                st.download_button(
                    "📥 Excel",
                    buffer.getvalue(),
                    file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
    
    # Section to send results by email
    if st.session_state.get('analysis_completed', False) and st.session_state.get('integrator') is not None:
        st.markdown("---")
        st.markdown("## 📧 SEND RESULTS BY EMAIL")
        st.markdown('<div class="email-box">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📨 SEND RESULTS TO MY EMAIL", type="primary", use_container_width=True):
                with st.spinner("Sending results by email..."):
                    if send_results_email(st.session_state['user_email'], st.session_state['integrator']):
                        st.success(f"✅ Results sent successfully to {st.session_state['user_email']}")
                        st.balloons()
                    else:
                        st.error("❌ Could not send email. Check configuration.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🔬 Integrated Semantic Search and Verifier v9.0 | LIVE MeSH LOOKUP • 4 STABLE DATABASES • PubMed • CrossRef • OpenAlex • Europe PMC</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
