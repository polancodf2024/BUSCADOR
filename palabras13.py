"""
PubMed AI Analyzer v8.4 - Human-Like Evidence Analysis Plus
MEJORAS v8.4:
- ENVÍO DE DOCX POR CORREO ELECTRÓNICO (configurado con secrets.toml)
- Alertas de calidad (semáforo)
- Detección de contradicciones entre estudios
- Línea de tiempo de publicaciones
- Tabla GRADE en DOCX
- Forest plot de tamaños del efecto
- Detección de sesgo de publicación
- Extracción PICO automática
- Resumen Ejecutivo + Análisis de Consenso
"""

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
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
import warnings
import hashlib
import json
from typing import List, Dict, Optional, Tuple
import pickle
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE SECRETS.TOML
# ============================================================================

def get_email_config():
    """Obtiene la configuración de email desde secrets.toml"""
    try:
        config = {
            'smtp_server': st.secrets.get("smtp_server", "smtp.gmail.com"),
            'smtp_port': int(st.secrets.get("smtp_port", 587)),
            'email_user': st.secrets.get("email_user", ""),
            'email_password': st.secrets.get("email_password", ""),
            'notification_email': st.secrets.get("notification_email", "")
        }
        return config
    except Exception as e:
        return {
            'smtp_server': "smtp.gmail.com",
            'smtp_port': 587,
            'email_user': "",
            'email_password': "",
            'notification_email': ""
        }


# ============================================================================
# NUEVA MEJORA v8.4: ENVÍO DE CORREO ELECTRÓNICO
# ============================================================================

class EmailSender:
    """Envía el documento DOCX por correo electrónico usando secrets.toml"""
    
    @staticmethod
    def send_docx_by_email(
        recipient_email: str,
        docx_bytes: BytesIO,
        filename: str,
        subject: str = None,
        body: str = None
    ) -> Tuple[bool, str]:
        """
        Envía un archivo DOCX por correo electrónico usando configuración de secrets.toml
        """
        config = get_email_config()
        
        # Verificar configuración
        if not config['email_user'] or not config['email_password']:
            return False, "❌ Configuración de correo no disponible. Verificar secrets.toml"
        
        if subject is None:
            subject = f"PubMed AI Analyzer - Análisis de Evidencia - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        if body is None:
            body = f"""
            Hola,
            
            Adjunto encontrará el análisis de evidencia generado por PubMed AI Analyzer v8.4.
            
            Archivo: {filename}
            Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            El documento incluye:
            - Resumen Ejecutivo y Análisis de Consenso
            - Tabla GRADE de evidencia
            - Forest plot de tamaños del efecto
            - Análisis de sesgo de publicación
            - Detección de contradicciones entre estudios
            - Línea de tiempo de publicaciones
            - Alertas de calidad metodológica
            - Flavors por relación con hipótesis
            - Clusters temáticos
            
            Este es un mensaje automático, por favor no responder a este correo.
            
            Saludos cordiales,
            PubMed AI Analyzer
            """
        
        # Crear mensaje
        msg = MIMEMultipart()
        msg['From'] = config['email_user']
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # Copia oculta al email de notificación si está configurado
        if config['notification_email']:
            msg['Bcc'] = config['notification_email']
        
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        # Adjuntar archivo
        try:
            docx_bytes.seek(0)
            attachment = MIMEBase('application', 'vnd.openxmlformats-officedocument.wordprocessingml.document')
            attachment.set_payload(docx_bytes.read())
            encoders.encode_base64(attachment)
            attachment.add_header(
                'Content-Disposition',
                f'attachment; filename={filename}'
            )
            msg.attach(attachment)
        except Exception as e:
            return False, f"❌ Error al adjuntar archivo: {str(e)}"
        
        # Enviar correo
        try:
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls()
                server.login(config['email_user'], config['email_password'])
                server.send_message(msg)
            
            # Mensaje de éxito con detalles
            success_msg = f"✅ Documento enviado correctamente a {recipient_email}"
            if config['notification_email']:
                success_msg += f" (copia a {config['notification_email']})"
            return True, success_msg
        except Exception as e:
            return False, f"❌ Error al enviar correo: {str(e)}"
    
    @staticmethod
    def get_email_config_status() -> Dict:
        """Verifica si la configuración de email está disponible"""
        config = get_email_config()
        
        return {
            'configured': bool(config['email_user'] and config['email_password']),
            'sender_email': config['email_user'] if config['email_user'] else "No configurado",
            'smtp_server': config['smtp_server'],
            'notification_email': config['notification_email'] if config['notification_email'] else "No configurado"
        }
    
    @staticmethod
    def test_connection() -> Tuple[bool, str]:
        """Prueba la conexión SMTP"""
        config = get_email_config()
        
        if not config['email_user'] or not config['email_password']:
            return False, "❌ Configuración incompleta en secrets.toml"
        
        try:
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls()
                server.login(config['email_user'], config['email_password'])
            return True, "✅ Conexión SMTP exitosa"
        except Exception as e:
            return False, f"❌ Error de conexión: {str(e)}"


# ============================================================================
# NUEVA MEJORA v8.3: ALERTAS DE CALIDAD (SISTEMA DE SEMÁFORO)
# ============================================================================

class QualityAlertSystem:
    """Sistema de alertas de calidad metodológica con semáforo"""
    
    @staticmethod
    def generate_alerts(articles: List[Dict]) -> Dict:
        """Genera alertas automáticas sobre limitaciones metodológicas"""
        articles = deduplicate_articles(articles)
        total = len(articles)
        
        alerts = {
            'critical': [],   # 🔴 Crítico
            'warning': [],    # 🟡 Advertencia
            'info': [],       # 🔵 Informativo
            'success': []     # 🟢 Éxito
        }
        
        # 1. Análisis de tipos de estudio
        case_reports = sum(1 for a in articles if a.get('evidence_level') == 'Reporte de caso')
        case_series = sum(1 for a in articles if a.get('evidence_level') == 'Serie de casos')
        rct = sum(1 for a in articles if a.get('evidence_level') == 'RCT')
        meta_analysis = sum(1 for a in articles if a.get('evidence_level') == 'Meta-análisis')
        
        if case_reports / total > 0.5:
            alerts['critical'].append({
                'message': f'⚠️ {case_reports}/{total} ({case_reports/total*100:.0f}%) estudios son reportes de caso (evidencia muy limitada)',
                'recommendation': 'Los resultados deben interpretarse con extrema cautela. Se necesitan estudios de mayor nivel evidencia.'
            })
        elif case_reports / total > 0.3:
            alerts['warning'].append({
                'message': f'📋 {case_reports}/{total} estudios son reportes de caso (>30% del total)',
                'recommendation': 'Considere que la evidencia es predominantemente observacional.'
            })
        
        if rct > 0:
            alerts['success'].append({
                'message': f'✅ {rct} ensayo(s) randomizado(s) incluido(s)',
                'recommendation': 'Fortaleza metodológica significativa.'
            })
        
        if meta_analysis > 0:
            alerts['success'].append({
                'message': f'📊 {meta_analysis} meta-análisis(es) incluido(s)',
                'recommendation': 'Alto nivel de evidencia sintetizada.'
            })
        
        # 2. Análisis de tamaños muestrales
        small_samples = [a for a in articles if a.get('sample_size') and a['sample_size'] < 50]
        large_samples = [a for a in articles if a.get('sample_size') and a['sample_size'] >= 200]
        
        if len(small_samples) > 0:
            alerts['warning'].append({
                'message': f'📏 {len(small_samples)} estudio(s) tienen N < 50 pacientes',
                'recommendation': 'Los resultados pueden no ser generalizables.'
            })
        
        if len(large_samples) >= 3:
            alerts['success'].append({
                'message': f'🎯 {len(large_samples)} estudio(s) con tamaño muestral ≥200',
                'recommendation': 'Buena potencia estadística.'
            })
        
        # 3. Análisis de calidad metodológica
        high_quality = [a for a in articles if (a.get('methodological_score') or 0) >= 70]
        low_quality = [a for a in articles if (a.get('methodological_score') or 0) < 50]
        
        if len(high_quality) >= len(articles) * 0.7:
            alerts['success'].append({
                'message': f'🏆 {len(high_quality)}/{len(articles)} estudios son de alta calidad metodológica (≥70/100)',
                'recommendation': 'La evidencia es metodológicamente robusta.'
            })
        elif len(low_quality) > len(articles) * 0.5:
            alerts['critical'].append({
                'message': f'⚠️ {len(low_quality)}/{len(articles)} estudios tienen baja calidad metodológica (<50/100)',
                'recommendation': 'Los hallazgos deben considerarse preliminares.'
            })
        
        # 4. Análisis de tamaños del efecto
        effect_studies = [a for a in articles if a.get('effect_value') and a['effect_value'] > 0]
        if len(effect_studies) >= 3:
            alerts['info'].append({
                'message': f'📈 {len(effect_studies)} estudios reportan tamaños del efecto (HR/OR/RR)',
                'recommendation': 'Posible realizar meta-análisis cuantitativo.'
            })
        
        # 5. Análisis de heterogeneidad
        positive_effect = [a for a in effect_studies if a.get('effect_value', 1) > 1]
        negative_effect = [a for a in effect_studies if a.get('effect_value', 1) < 1]
        
        if len(positive_effect) > 0 and len(negative_effect) > 0:
            alerts['warning'].append({
                'message': f'🔄 Heterogeneidad detectada: {len(positive_effect)} estudios efecto positivo, {len(negative_effect)} efecto negativo',
                'recommendation': 'Existe inconsistencia en la dirección del efecto.'
            })
        
        # Determinar nivel de alerta general
        if alerts['critical']:
            overall_level = "🔴 CRÍTICO"
            overall_message = "Existen limitaciones metodológicas graves que afectan la confianza en los resultados."
        elif alerts['warning']:
            overall_level = "🟡 ADVERTENCIA"
            overall_message = "Se identificaron limitaciones metodológicas que deben considerarse al interpretar los resultados."
        elif alerts['success']:
            overall_level = "🟢 FAVORABLE"
            overall_message = "La evidencia muestra buena calidad metodológica."
        else:
            overall_level = "🔵 INFORMATIVO"
            overall_message = "Revise las recomendaciones para una interpretación adecuada."
        
        return {
            'alerts': alerts,
            'overall_level': overall_level,
            'overall_message': overall_message,
            'total_articles': total
        }


# ============================================================================
# NUEVA MEJORA v8.3: DETECCIÓN DE CONTRADICCIONES ENTRE ESTUDIOS
# ============================================================================

class ContradictionDetector:
    """Detecta contradicciones explícitas entre estudios"""
    
    @staticmethod
    def find_contradictions(articles: List[Dict]) -> List[Dict]:
        """Encuentra pares de artículos con conclusiones opuestas"""
        articles = deduplicate_articles(articles)
        contradictions = []
        
        for i, a1 in enumerate(articles):
            for a2 in articles[i+1:]:
                conflict = None
                
                if a1.get('effect_value') and a2.get('effect_value'):
                    if a1['effect_value'] > 1 and a2['effect_value'] < 1:
                        conflict = {
                            'type': 'effect_direction',
                            'description': f"Dirección del efecto opuesta",
                            'detail': f"{a1.get('authors', 'Estudio A')[:30]} reporta {a1.get('effect_type', 'HR')}={a1['effect_value']:.2f} (>1) vs {a2.get('authors', 'Estudio B')[:30]} reporta {a2.get('effect_type', 'HR')}={a2['effect_value']:.2f} (<1)"
                        }
                    elif a1['effect_value'] < 1 and a2['effect_value'] > 1:
                        conflict = {
                            'type': 'effect_direction',
                            'description': f"Dirección del efecto opuesta",
                            'detail': f"{a1.get('authors', 'Estudio A')[:30]} reporta {a1.get('effect_type', 'HR')}={a1['effect_value']:.2f} (<1) vs {a2.get('authors', 'Estudio B')[:30]} reporta {a2.get('effect_type', 'HR')}={a2['effect_value']:.2f} (>1)"
                        }
                
                if not conflict:
                    rel1 = a1.get('hypothesis_relation', 'unrelated')
                    rel2 = a2.get('hypothesis_relation', 'unrelated')
                    if (rel1 == 'confirms' and rel2 == 'contradicts') or (rel1 == 'contradicts' and rel2 == 'confirms'):
                        conflict = {
                            'type': 'hypothesis_relation',
                            'description': f"Relación con hipótesis opuesta",
                            'detail': f"{a1.get('authors', 'Estudio A')[:30]}: {rel1} vs {a2.get('authors', 'Estudio B')[:30]}: {rel2}"
                        }
                
                if conflict:
                    contradictions.append({
                        'article1': a1,
                        'article2': a2,
                        'conflict': conflict
                    })
        
        return contradictions
    
    @staticmethod
    def generate_contradiction_summary(contradictions: List[Dict]) -> str:
        """Genera resumen de contradicciones encontradas"""
        if not contradictions:
            return "✅ **No se detectaron contradicciones significativas** entre los estudios analizados."
        
        summary = f"⚠️ **Se detectaron {len(contradictions)} contradicción(es) entre estudios:**\n\n"
        
        for i, contra in enumerate(contradictions[:5], 1):
            conflict = contra['conflict']
            summary += f"{i}. **{conflict['description']}**\n"
            summary += f"   {conflict['detail']}\n\n"
        
        if len(contradictions) > 5:
            summary += f"... y {len(contradictions) - 5} contradicción(es) adicional(es).\n"
        
        summary += "\n**Recomendación:** Revisar los artículos en conflicto para identificar fuentes de heterogeneidad."
        
        return summary


# ============================================================================
# NUEVA MEJORA v8.3: LÍNEA DE TIEMPO DE PUBLICACIONES
# ============================================================================

class TemporalConsensusAnalyzer:
    """Analiza la evolución temporal del consenso científico"""
    
    @staticmethod
    def analyze_temporal_consensus(articles: List[Dict]) -> Dict:
        articles = deduplicate_articles(articles)
        
        years_data = {}
        for art in articles:
            year = art.get('pubdate', '')[:4]
            if year and year.isdigit() and 1990 <= int(year) <= datetime.now().year + 1:
                if year not in years_data:
                    years_data[year] = {
                        'confirms': 0, 'contradicts': 0, 'nuances': 0, 
                        'extends': 0, 'total': 0, 'avg_quality': []
                    }
                years_data[year]['total'] += 1
                relation = art.get('hypothesis_relation', 'unrelated')
                if relation == 'confirms':
                    years_data[year]['confirms'] += 1
                elif relation == 'contradicts':
                    years_data[year]['contradicts'] += 1
                elif relation == 'nuances':
                    years_data[year]['nuances'] += 1
                elif relation == 'extends':
                    years_data[year]['extends'] += 1
                
                quality = art.get('quality_score', 0)
                if quality:
                    years_data[year]['avg_quality'].append(quality)
        
        for year, data in years_data.items():
            if data['total'] > 0:
                data['confirms_pct'] = (data['confirms'] / data['total']) * 100
                data['contradicts_pct'] = (data['contradicts'] / data['total']) * 100
                data['avg_quality_score'] = sum(data['avg_quality']) / len(data['avg_quality']) if data['avg_quality'] else 0
        
        years_sorted = sorted(years_data.keys())
        if len(years_sorted) >= 3:
            recent_years = years_sorted[-3:]
            recent_avg_confirms = sum(years_data[y]['confirms_pct'] for y in recent_years) / 3
            old_avg_confirms = sum(years_data[y]['confirms_pct'] for y in years_sorted[:3]) / 3
            
            if recent_avg_confirms > old_avg_confirms + 15:
                trend = "📈 **Tendencia ALCISTA** - Aumenta la confirmación de la hipótesis"
            elif recent_avg_confirms < old_avg_confirms - 15:
                trend = "📉 **Tendencia BAJISTA** - Disminuye la confirmación de la hipótesis"
            else:
                trend = "➡️ **Tendencia ESTABLE** - El consenso se ha mantenido constante"
        else:
            trend = "⚠️ **Datos insuficientes** para determinar tendencia"
        
        return {
            'years_data': years_data,
            'trend': trend,
            'first_year': min(years_sorted) if years_sorted else None,
            'last_year': max(years_sorted) if years_sorted else None,
            'total_years': len(years_sorted)
        }
    
    @staticmethod
    def create_temporal_plot(temporal_data: Dict) -> Optional[BytesIO]:
        if not temporal_data or not temporal_data.get('years_data'):
            return None
        
        try:
            import matplotlib.pyplot as plt
            
            years = sorted(temporal_data['years_data'].keys())
            confirms_pct = [temporal_data['years_data'][y]['confirms_pct'] for y in years]
            contradicts_pct = [temporal_data['years_data'][y]['contradicts_pct'] for y in years]
            total_studies = [temporal_data['years_data'][y]['total'] for y in years]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            ax1.plot(years, confirms_pct, 'g-o', linewidth=2, markersize=8, label='Confirman (%)')
            ax1.plot(years, contradicts_pct, 'r-s', linewidth=2, markersize=8, label='Contradicen (%)')
            ax1.set_xlabel('Año de publicación')
            ax1.set_ylabel('Porcentaje de estudios (%)')
            ax1.set_title('Evolución del consenso científico por año')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 100)
            
            for i, (year, n) in enumerate(zip(years, total_studies)):
                ax1.annotate(f'n={n}', (year, confirms_pct[i]), 
                            textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
            
            ax2.bar(years, confirms_pct, label='Confirman', color='green', alpha=0.7)
            ax2.bar(years, contradicts_pct, bottom=confirms_pct, label='Contradicen', color='red', alpha=0.7)
            
            remaining = [100 - confirms_pct[i] - contradicts_pct[i] for i in range(len(years))]
            ax2.bar(years, remaining, bottom=[confirms_pct[i] + contradicts_pct[i] for i in range(len(years))], 
                   label='Matizan/Extienden', color='orange', alpha=0.5)
            
            ax2.set_xlabel('Año de publicación')
            ax2.set_ylabel('Distribución (%)')
            ax2.set_title('Distribución de relaciones con hipótesis por año')
            ax2.legend(loc='upper right')
            ax2.set_ylim(0, 100)
            
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            return buf
        except Exception as e:
            print(f"Error en gráfico temporal: {e}")
            return None
    
    @staticmethod
    def generate_temporal_summary(temporal_data: Dict) -> str:
        if not temporal_data or not temporal_data.get('years_data'):
            return "No hay suficientes datos para análisis temporal."
        
        years_data = temporal_data['years_data']
        years_sorted = sorted(years_data.keys())
        
        if len(years_sorted) < 2:
            return f"Estudios concentrados en un solo año ({years_sorted[0]})."
        
        summary = f"**📅 Análisis temporal ({temporal_data['first_year']} - {temporal_data['last_year']})**\n\n"
        summary += f"• Período analizado: {temporal_data['total_years']} años\n"
        summary += f"• {temporal_data['trend']}\n\n"
        
        max_year = max(years_data.keys(), key=lambda y: years_data[y]['total'])
        summary += f"• **Mayor producción:** {max_year} ({years_data[max_year]['total']} estudios)\n"
        
        max_confirm_year = max(years_data.keys(), key=lambda y: years_data[y]['confirms_pct'])
        summary += f"• **Mayor consenso:** {max_confirm_year} ({years_data[max_confirm_year]['confirms_pct']:.0f}% confirmación)\n"
        
        return summary


# ============================================================================
# NUEVA MEJORA v8.2: EXTRACCIÓN PICO
# ============================================================================

class PICOExtractor:
    @staticmethod
    def extract_population(text: str) -> str:
        text_lower = text.lower()
        patterns = [
            r'patients? with ([^.!]+)',
            r'subjects? with ([^.!]+)',
            r'population of ([^.!]+)',
            r'enrolled ([^.!]+ patients?)',
            r'study included ([^.!]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1).strip()[:100]
        return "No especificada"
    
    @staticmethod
    def extract_intervention(text: str) -> str:
        text_lower = text.lower()
        patterns = [
            r'(?:treated with|received|underwent) ([^.!]+)',
            r'intervention: ([^.!]+)',
            r'exposed to ([^.!]+)',
            r'(?:surgical|medical) ([^.!]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1).strip()[:100]
        return "No especificada"
    
    @staticmethod
    def extract_comparison(text: str) -> str:
        text_lower = text.lower()
        patterns = [
            r'compared to ([^.!]+)',
            r'vs\.? ([^.!]+)',
            r'versus ([^.!]+)',
            r'compared with ([^.!]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1).strip()[:100]
        return "No especificada"
    
    @staticmethod
    def extract_outcomes(text: str) -> List[str]:
        text_lower = text.lower()
        outcome_keywords = [
            'mortality', 'death', 'survival', 'rupture', 'bleeding', 'hemorrhage',
            'stroke', 'reinfarction', 'complication', 'heart failure', 'hf',
            'mace', 'major adverse', 'recovery', 'improvement', 'remodeling'
        ]
        found = [kw for kw in outcome_keywords if kw in text_lower]
        return found[:3] if found else ["No especificado"]
    
    @classmethod
    def extract_all(cls, title: str, abstract: str) -> Dict:
        full_text = f"{title} {abstract if abstract else ''}"
        return {
            'population': cls.extract_population(full_text),
            'intervention': cls.extract_intervention(full_text),
            'comparison': cls.extract_comparison(full_text),
            'outcomes': cls.extract_outcomes(full_text),
            'pico_string': f"P: {cls.extract_population(full_text)[:50]} | I: {cls.extract_intervention(full_text)[:30]} | O: {', '.join(cls.extract_outcomes(full_text)[:2])}"
        }


# ============================================================================
# DEDUPLICACIÓN DE ARTÍCULOS
# ============================================================================

def deduplicate_articles(articles: List[Dict]) -> List[Dict]:
    seen_pmids = set()
    unique_articles = []
    for article in articles:
        pmid = article.get('pmid', '')
        if pmid and pmid not in seen_pmids:
            seen_pmids.add(pmid)
            unique_articles.append(article)
        elif not pmid:
            title = article.get('title', '')[:100]
            if title not in seen_pmids:
                seen_pmids.add(title)
                unique_articles.append(article)
    return unique_articles


# ============================================================================
# ANÁLISIS DE CONSENSO Y RESUMEN EJECUTIVO
# ============================================================================

class ConsensusAnalyzer:
    @staticmethod
    def calculate_consensus(articles: List[Dict]) -> Dict:
        if not articles:
            return {'consensus_level': 'Sin datos', 'percentage': 0, 'details': {}}
        
        articles = deduplicate_articles(articles)
        relation_counts = Counter()
        for a in articles:
            relation = a.get('hypothesis_relation', 'unrelated')
            relation_counts[relation] += 1
        
        total = len(articles)
        confirms_pct = (relation_counts.get('confirms', 0) / total) * 100
        contradicts_pct = (relation_counts.get('contradicts', 0) / total) * 100
        nuances_pct = (relation_counts.get('nuances', 0) / total) * 100
        extends_pct = (relation_counts.get('extends', 0) / total) * 100
        
        if confirms_pct >= 70:
            consensus_level = "MUY ALTO"
            consensus_color = "🟢"
            consensus_message = "La evidencia es sólida y consistente. La hipótesis está ampliamente respaldada."
        elif confirms_pct >= 50:
            consensus_level = "MODERADO"
            consensus_color = "🟡"
            consensus_message = "Existe respaldo mayoritario, pero hay evidencia que matiza o contradice parcialmente."
        elif confirms_pct >= 30:
            consensus_level = "BAJO"
            consensus_color = "🟠"
            consensus_message = "La evidencia es mixta. Se necesitan más estudios."
        else:
            consensus_level = "MUY BAJO / CONTROVERSIA"
            consensus_color = "🔴"
            consensus_message = "Existe controversia significativa."
        
        high_quality_confirms = sum(1 for a in articles 
                                    if a.get('hypothesis_relation') == 'confirms' 
                                    and (a.get('evidence_score') or 0) >= 80)
        
        return {
            'consensus_level': consensus_level,
            'consensus_color': consensus_color,
            'consensus_message': consensus_message,
            'confirms_percentage': confirms_pct,
            'contradicts_percentage': contradicts_pct,
            'nuances_percentage': nuances_pct,
            'extends_percentage': extends_pct,
            'high_quality_confirms': high_quality_confirms,
            'total_articles': total,
            'relation_counts': dict(relation_counts)
        }
    
    @staticmethod
    def generate_executive_summary(articles: List[Dict], hypothesis: str, query: str) -> str:
        articles = deduplicate_articles(articles)
        consensus = ConsensusAnalyzer.calculate_consensus(articles)
        
        avg_quality = np.mean([a.get('quality_score', 0) or 0 for a in articles])
        avg_evidence = np.mean([a.get('evidence_score', 0) or 0 for a in articles])
        
        best_article = max(articles, key=lambda x: x.get('evidence_score', 0) or 0)
        best_evidence = best_article.get('evidence_score', 0)
        best_design = best_article.get('study_design', 'Estudio')
        
        years = [a.get('pubdate', '')[:4] for a in articles if a.get('pubdate') and len(a.get('pubdate', '')) >= 4]
        year_counts = Counter(years)
        
        summary = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         📋 RESUMEN EJECUTIVO                                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝

🔬 HIPÓTESIS EVALUADA:
   "{hypothesis[:200]}..."

📊 NIVEL DE CONSENSO: {consensus['consensus_color']} {consensus['consensus_level']}
   {consensus['consensus_message']}

📈 DISTRIBUCIÓN DE EVIDENCIA:
   • Confirman la hipótesis: {consensus['confirms_percentage']:.1f}% ({consensus['relation_counts'].get('confirms', 0)} estudios)
   • Matizan la hipótesis: {consensus['nuances_percentage']:.1f}% ({consensus['relation_counts'].get('nuances', 0)} estudios)
   • Extienden la hipótesis: {consensus['extends_percentage']:.1f}% ({consensus['relation_counts'].get('extends', 0)} estudios)
   • Contradicen la hipótesis: {consensus['contradicts_percentage']:.1f}% ({consensus['relation_counts'].get('contradicts', 0)} estudios)

⭐ CALIDAD DE LA EVIDENCIA:
   • Calidad metodológica promedio: {avg_quality:.0f}/100
   • Nivel de evidencia promedio: {avg_evidence:.0f}/100
   • Estudios de alta calidad que CONFIRMAN: {consensus['high_quality_confirms']}

🏆 ESTUDIO DESTACADO:
   • {best_article.get('authors', 'Autor')[:80]} ({best_article.get('pubdate', 's.f.')[:4]})
   • Diseño: {best_design}
   • Score evidencia: {best_evidence}/100

"""
        summary += "\n💡 RECOMENDACIÓN CLÍNICA:\n   "
        if consensus['consensus_level'] in ["MUY ALTO", "MODERADO"] and consensus['confirms_percentage'] >= 50:
            summary += "La evidencia respalda la hipótesis. Los hallazgos son consistentes y aplicables a la práctica clínica."
        elif consensus['consensus_level'] == "BAJO":
            summary += "La evidencia es mixta. Se recomienda cautela en la interpretación."
        else:
            summary += "Existe controversia significativa. No se puede establecer una conclusión firme."
        
        return summary


# ============================================================================
# CONFIGURACIÓN DE EMBEDDINGS
# ============================================================================

AI_EMBEDDINGS_AVAILABLE = False
BIOMED_EMBEDDER = None
FALLBACK_EMBEDDER = None
USE_FALLBACK = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        
        with st.spinner("🔄 Loading embeddings model..."):
            try:
                BIOMED_EMBEDDER = SentenceTransformer(
                    'pritamdeka/S-Biomed-Roberta-snli-multinli-stsb',
                    device='cpu'
                )
                AI_EMBEDDINGS_AVAILABLE = True
                USE_FALLBACK = False
                st.success("✅ BioBERT embeddings available")
            except:
                try:
                    FALLBACK_EMBEDDER = SentenceTransformer(
                        'all-MiniLM-L6-v2',
                        device='cpu'
                    )
                    AI_EMBEDDINGS_AVAILABLE = True
                    USE_FALLBACK = True
                    st.warning("⚠️ BioBERT unavailable, using SBERT")
                except:
                    AI_EMBEDDINGS_AVAILABLE = False
    except ImportError:
        AI_EMBEDDINGS_AVAILABLE = False
        
except Exception as e:
    print(f"⚠️ Some ML libraries not available: {e}")
    AI_EMBEDDINGS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    st.warning("⚠️ matplotlib no instalado")

st.set_page_config(
    page_title="PubMed AI Analyzer v8.4",
    page_icon="🧠",
    layout="wide"
)


# ============================================================================
# CACHÉ DE EMBEDDINGS
# ============================================================================

class EmbeddingCache:
    def __init__(self, cache_dir="./embedding_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    def get(self, text: str) -> Optional[np.ndarray]:
        hash_key = self._get_hash(text)
        cache_file = os.path.join(self.cache_dir, f"{hash_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return None
    
    def set(self, text: str, embedding: np.ndarray):
        hash_key = self._get_hash(text)
        cache_file = os.path.join(self.cache_dir, f"{hash_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except:
            pass

embedding_cache = EmbeddingCache()


# ============================================================================
# SISTEMA DE PONDERACIÓN POR NIVEL DE EVIDENCIA
# ============================================================================

class EvidenceHierarchy:
    EVIDENCE_LEVELS = {
        'meta-analysis': {'level': 1, 'score': 100, 'name': 'Meta-análisis'},
        'systematic_review': {'level': 1, 'score': 100, 'name': 'Revisión sistemática'},
        'rct': {'level': 1, 'score': 95, 'name': 'RCT'},
        'randomized_trial': {'level': 1, 'score': 95, 'name': 'Ensayo randomizado'},
        'prospective_cohort': {'level': 2, 'score': 80, 'name': 'Cohorte prospectivo'},
        'retrospective_cohort': {'level': 3, 'score': 60, 'name': 'Cohorte retrospectivo'},
        'case_control': {'level': 3, 'score': 55, 'name': 'Caso-control'},
        'case_series': {'level': 4, 'score': 35, 'name': 'Serie de casos'},
        'case_report': {'level': 5, 'score': 20, 'name': 'Reporte de caso'},
        'observational': {'level': 3, 'score': 50, 'name': 'Observacional'},
    }
    
    @classmethod
    def classify_study_design(cls, title: str, abstract: str, study_types: str) -> Tuple[str, int, int]:
        text = f"{title} {abstract} {study_types}".lower()
        
        patterns = {
            'meta-analysis': ['meta-analysis', 'meta analysis', 'systematic review'],
            'rct': ['randomized controlled trial', 'randomised controlled trial', 'rct'],
            'prospective_cohort': ['prospective cohort', 'prospective study'],
            'retrospective_cohort': ['retrospective cohort', 'retrospective study'],
            'case_control': ['case-control', 'case control'],
            'case_series': ['case series'],
            'case_report': ['case report'],
        }
        
        for design, keywords in patterns.items():
            if any(kw in text for kw in keywords):
                info = cls.EVIDENCE_LEVELS.get(design, cls.EVIDENCE_LEVELS['observational'])
                return info['name'], info['level'], info['score']
        
        return 'No clasificado', 5, 10
    
    @classmethod
    def extract_sample_size(cls, text: str) -> Optional[int]:
        patterns = [
            r'n\s*[=:]\s*(\d+)',
            r'sample size\s*[=:]\s*(\d+)',
            r'patients?\s*[=:]\s*(\d+)',
            r'(\d+)\s*patients?',
        ]
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        return None


class NumericalExtractor:
    @staticmethod
    def extract_all_numbers(text: str) -> Dict:
        results = {'hr': [], 'rr': [], 'or': [], 'p_values': [], 'n': None}
        
        hr_pattern = r'HR[\s]*[=:]?[\s]*([0-9.]+)[\s]*\(([0-9.]+)[-\s]+([0-9.]+)\)'
        for match in re.finditer(hr_pattern, text, re.IGNORECASE):
            results['hr'].append({
                'value': float(match.group(1)),
                'ci_lower': float(match.group(2)),
                'ci_upper': float(match.group(3))
            })
        
        rr_pattern = r'RR[\s]*[=:]?[\s]*([0-9.]+)[\s]*\(([0-9.]+)[-\s]+([0-9.]+)\)'
        for match in re.finditer(rr_pattern, text, re.IGNORECASE):
            results['rr'].append({
                'value': float(match.group(1)),
                'ci_lower': float(match.group(2)),
                'ci_upper': float(match.group(3))
            })
        
        or_pattern = r'OR[\s]*[=:]?[\s]*([0-9.]+)[\s]*\(([0-9.]+)[-\s]+([0-9.]+)\)'
        for match in re.finditer(or_pattern, text, re.IGNORECASE):
            results['or'].append({
                'value': float(match.group(1)),
                'ci_lower': float(match.group(2)),
                'ci_upper': float(match.group(3))
            })
        
        p_pattern = r'p[\s]*[=<>]?\s*([0-9.]+[e-]?[0-9]*)'
        for match in re.finditer(p_pattern, text, re.IGNORECASE):
            try:
                results['p_values'].append(float(match.group(1)))
            except:
                pass
        
        results['n'] = EvidenceHierarchy.extract_sample_size(text)
        return results
    
    @staticmethod
    def get_best_effect_size(numbers: Dict) -> Optional[Dict]:
        for effect_type in ['hr', 'or', 'rr']:
            if numbers.get(effect_type):
                return {
                    'type': effect_type.upper(),
                    'value': numbers[effect_type][0]['value'],
                    'ci_lower': numbers[effect_type][0]['ci_lower'],
                    'ci_upper': numbers[effect_type][0]['ci_upper']
                }
        return None


class HypothesisRelationClassifier:
    @staticmethod
    def classify(article_text: str, hypothesis: str, embedding_model=None) -> Dict:
        if not article_text or not hypothesis:
            return {'relation': 'unrelated', 'confidence': 0}
        
        text_lower = article_text.lower()
        
        confirms_keywords = ['confirms', 'demonstrates', 'shows', 'consistent with', 'supports', 'validates']
        nuances_keywords = ['however', 'although', 'but', 'except', 'whereas', 'while']
        extends_keywords = ['furthermore', 'moreover', 'additionally', 'also', 'extends', 'expands']
        contradicts_keywords = ['contrary to', 'contradicts', 'disagrees', 'opposite', 'unexpectedly']
        
        score_confirms = sum(1 for kw in confirms_keywords if kw in text_lower)
        score_nuances = sum(1 for kw in nuances_keywords if kw in text_lower)
        score_extends = sum(1 for kw in extends_keywords if kw in text_lower)
        score_contradicts = sum(1 for kw in contradicts_keywords if kw in text_lower)
        
        embedding_score = 0.5
        if embedding_model and AI_EMBEDDINGS_AVAILABLE:
            try:
                text_emb = embedding_model.encode([article_text[:1000]])[0]
                hyp_emb = embedding_model.encode([hypothesis])[0]
                embedding_score = cosine_similarity([text_emb], [hyp_emb])[0][0]
            except:
                pass
        
        scores = {
            'confirms': score_confirms * 0.3 + embedding_score * 0.7,
            'nuances': score_nuances * 0.5,
            'extends': score_extends * 0.4,
            'contradicts': score_contradicts * 0.6
        }
        
        best_relation = max(scores, key=scores.get)
        confidence = min(scores[best_relation], 1.0)
        
        if confidence < 0.2:
            best_relation = 'unrelated'
        
        return {'relation': best_relation, 'confidence': confidence}


# ============================================================================
# CHECKLIST METODOLÓGICO
# ============================================================================

class MethodologicalChecklist:
    @staticmethod
    def check_article(article: Dict) -> Tuple[Dict, float]:
        text = f"{article.get('title', '')} {article.get('abstract', '')}".lower()
        
        checklist = {
            'Prospectivo': bool(re.search(r'prospective|longitudinal|cohort', text)),
            'Aleatorizado': bool(re.search(r'randomized|randomised', text)),
            'Cegamiento': bool(re.search(r'blind|masked|double-blind', text)),
            'Multicéntrico': bool(re.search(r'multicenter|multi-center', text)),
            'Análisis multivariado': bool(re.search(r'multivariable|multivariate|adjusted|cox', text)),
            'Registro clínico': bool(re.search(r'clinicaltrials\.gov|nct\d+', text)),
        }
        
        score = sum(checklist.values()) / len(checklist) * 100
        return checklist, score


# ============================================================================
# EXPORTACIÓN RIS Y BibTeX
# ============================================================================

class ReferenceExporter:
    @staticmethod
    def to_ris(articles: List[Dict]) -> str:
        articles = deduplicate_articles(articles)
        ris_lines = []
        for a in articles:
            ris_lines.append("TY  - JOUR")
            ris_lines.append(f"AU  - {a.get('authors', '')}")
            ris_lines.append(f"TI  - {a.get('title', '')}")
            ris_lines.append(f"JO  - {a.get('journal', '')}")
            ris_lines.append(f"PY  - {a.get('pubdate', '')[:4]}")
            ris_lines.append(f"ID  - {a.get('pmid', '')}")
            ris_lines.append("ER  - ")
            ris_lines.append("")
        return "\n".join(ris_lines)
    
    @staticmethod
    def to_bibtex(articles: List[Dict]) -> str:
        articles = deduplicate_articles(articles)
        bib_lines = []
        for i, a in enumerate(articles):
            pmid = a.get('pmid', f'unknown_{i}')
            bib_lines.append(f"@article{{pmid_{pmid},")
            bib_lines.append(f"  title = {{{a.get('title', '')}}},")
            bib_lines.append(f"  journal = {{{a.get('journal', '')}}},")
            bib_lines.append(f"  year = {{{a.get('pubdate', '')[:4]}}},")
            bib_lines.append(f"  pmid = {{{pmid}}}")
            bib_lines.append("}")
            bib_lines.append("")
        return "\n".join(bib_lines)


# ============================================================================
# GRÁFICOS
# ============================================================================

def plot_consensus_gauge(consensus: Dict) -> Optional[BytesIO]:
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    fig, ax = plt.subplots(figsize=(6, 4))
    confirms_pct = consensus.get('confirms_percentage', 0)
    
    if confirms_pct >= 70:
        color = '#2ecc71'
    elif confirms_pct >= 50:
        color = '#f39c12'
    elif confirms_pct >= 30:
        color = '#e67e22'
    else:
        color = '#e74c3c'
    
    ax.barh(['Consenso'], [confirms_pct], color=color, height=0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Porcentaje de estudios que confirman la hipótesis (%)')
    ax.set_title(f'Nivel de Consenso: {consensus.get("consensus_level", "ND")}')
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
    ax.text(confirms_pct + 2, 0, f'{confirms_pct:.0f}%', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf


# ============================================================================
# FUNCIONES DE UTILIDAD PARA PubMed
# ============================================================================

def make_request_with_retry(url, params, max_retries=5, initial_delay=5):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 429:
                wait_time = delay * (2 ** attempt)
                st.warning(f"⚠️ Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            return response
        except:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
    return None


def search_pubmed_complete(query, max_articles=3000, progress_callback=None):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = f"{base_url}esearch.fcgi"
    
    try:
        st.info("🔍 Buscando artículos en PubMed...")
        params = {"db": "pubmed", "term": query, "retmax": 1, "retmode": "xml"}
        response = make_request_with_retry(search_url, params)
        if response is None:
            return [], 0
        
        root = ElementTree.fromstring(response.content)
        total_count = int(root.find(".//Count").text) if root.find(".//Count") is not None else 0
        st.info(f"📊 PubMed encontró {total_count} artículos")
        
        if total_count == 0:
            return [], 0
        
        articles_to_fetch = min(total_count, max_articles)
        st.info(f"📊 Se procesarán {articles_to_fetch} artículos")
        
        all_ids = []
        retstart = 0
        batch_size = 10000
        progress_bar = st.progress(0)
        
        while retstart < articles_to_fetch:
            remaining = articles_to_fetch - retstart
            current_batch = min(batch_size, remaining)
            
            fetch_params = {"db": "pubmed", "term": query, "retstart": retstart, "retmax": current_batch, "retmode": "xml"}
            
            try:
                fetch_response = make_request_with_retry(search_url, fetch_params)
                if fetch_response is None:
                    break
                
                fetch_root = ElementTree.fromstring(fetch_response.content)
                batch_ids = [id_elem.text for id_elem in fetch_root.findall(".//Id")]
                
                if not batch_ids:
                    break
                
                all_ids.extend(batch_ids)
                progress_bar.progress(min(len(all_ids) / articles_to_fetch, 1.0))
                if progress_callback:
                    progress_callback(len(all_ids), articles_to_fetch)
                retstart += current_batch
                time.sleep(0.5)
            except Exception as e:
                st.error(f"Error: {e}")
                break
        
        progress_bar.empty()
        st.success(f"✅ Recuperados {len(all_ids)} IDs")
        return all_ids, len(all_ids)
    except Exception as e:
        st.error(f"Error: {e}")
        return [], 0


@st.cache_data(ttl=3600, show_spinner=False)
def get_abstract(pmid):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    fetch_url = f"{base_url}efetch.fcgi"
    params = {"db": "pubmed", "id": pmid, "retmode": "xml", "rettype": "abstract"}
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
    except:
        return None


def extract_article_info(doc_sum):
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
    return article


def get_embedder():
    if BIOMED_EMBEDDER is not None:
        return BIOMED_EMBEDDER
    elif FALLBACK_EMBEDDER is not None:
        return FALLBACK_EMBEDDER
    return None


def extract_all_outcomes(text):
    if not text:
        return []
    text_lower = text.lower()
    outcomes = set()
    indicators = ['mortality', 'death', 'survival', 'rupture', 'bleeding', 'hemorrhage', 
                  'stroke', 'reinfarction', 'complication', 'heart failure', 'mace']
    for indicator in indicators:
        if indicator in text_lower:
            outcomes.add(indicator)
    return list(outcomes)


def enhanced_article_analysis(title: str, abstract: str, hypothesis: str = "") -> Dict:
    full_text = f"{title} {abstract if abstract else ''}"
    
    study_design, evidence_level, evidence_score = EvidenceHierarchy.classify_study_design(title, abstract, "")
    
    numbers = NumericalExtractor.extract_all_numbers(full_text)
    sample_size = numbers.get('n')
    best_effect = NumericalExtractor.get_best_effect_size(numbers)
    
    embedder = get_embedder()
    relation_info = HypothesisRelationClassifier.classify(full_text, hypothesis, embedder)
    
    all_outcomes = extract_all_outcomes(full_text)
    pico = PICOExtractor.extract_all(title, abstract)
    
    article_dict = {'title': title, 'abstract': abstract}
    checklist, methodological_score = MethodologicalChecklist.check_article(article_dict)
    
    quality_score = (evidence_score + methodological_score) / 2
    
    if sample_size and sample_size > 100:
        quality_score += 5
    if best_effect:
        if best_effect['value'] < 0.7 or best_effect['value'] > 1.5:
            quality_score += 10
    quality_score = min(quality_score, 100)
    
    return {
        'study_design': study_design,
        'evidence_level': evidence_level,
        'evidence_score': evidence_score,
        'sample_size': sample_size,
        'effect_type': best_effect['type'] if best_effect else '',
        'effect_value': best_effect['value'] if best_effect else 0,
        'effect_ci_lower': best_effect['ci_lower'] if best_effect else 0,
        'effect_ci_upper': best_effect['ci_upper'] if best_effect else 0,
        'p_values': numbers['p_values'][:3] if numbers['p_values'] else [],
        'hypothesis_relation': relation_info['relation'],
        'relation_confidence': relation_info['confidence'],
        'all_outcomes': all_outcomes,
        'quality_score': quality_score,
        'methodological_score': methodological_score,
        'methodological_checklist': checklist,
        'pico_population': pico['population'],
        'pico_intervention': pico['intervention'],
        'pico_comparison': pico['comparison'],
        'pico_outcomes': ', '.join(pico['outcomes']),
        'pico_string': pico['pico_string']
    }


def fetch_articles_details(id_list, query_terms, hypothesis_terms, hypothesis="", block_number=0, progress_callback=None):
    if not id_list:
        return []
    
    total_to_process = len(id_list)
    batch_size = 30
    num_batches = math.ceil(total_to_process / batch_size)
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    articles = []
    
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, total_to_process)
        batch_ids = id_list[start_idx:end_idx]
        
        summary_params = {"db": "pubmed", "id": ",".join(batch_ids), "retmode": "xml"}
        
        for retry in range(3):
            try:
                summary_response = make_request_with_retry(f"{base_url}esummary.fcgi", summary_params)
                if summary_response is None:
                    if retry < 2:
                        time.sleep(5)
                        continue
                    else:
                        raise Exception("Max retries exceeded")
                
                summary_root = ElementTree.fromstring(summary_response.content)
                
                for j, doc_sum in enumerate(summary_root.findall(".//DocSum")):
                    overall_idx = start_idx + j
                    if progress_callback:
                        progress_callback(overall_idx + 1, total_to_process)
                    
                    article = extract_article_info(doc_sum)
                    abstract = get_abstract(article["pmid"])
                    article["abstract"] = abstract if abstract else "Not available"
                    
                    enhanced = enhanced_article_analysis(article["title"], abstract, hypothesis)
                    article.update(enhanced)
                    article["block_number"] = block_number
                    articles.append(article)
                break
            except Exception as e:
                if retry == 2:
                    st.warning(f"Error en lote {batch_num + 1}: {str(e)[:50]}")
                else:
                    time.sleep(5)
        
        time.sleep(1)
    
    return articles


def calculate_relevance_to_hypothesis(articles, hypothesis):
    embedder = get_embedder()
    if not embedder or not AI_EMBEDDINGS_AVAILABLE:
        for article in articles:
            article['relevance_score'] = 0.5
        return articles
    
    try:
        hypothesis_embedding = embedder.encode([hypothesis[:500]])[0]
        for article in articles:
            text = f"{article.get('title', '')} {article.get('abstract', '')}"
            if text and len(text) > 50:
                article_embedding = embedder.encode([text[:1000]])[0]
                similarity = cosine_similarity([hypothesis_embedding], [article_embedding])[0][0]
                article['relevance_score'] = float(similarity)
            else:
                article['relevance_score'] = 0.0
    except Exception as e:
        st.warning(f"Error en relevancia: {e}")
        for article in articles:
            article['relevance_score'] = 0.5
    return articles


def filter_articles_by_relevance(articles, relevance_threshold):
    filtered = [a for a in articles if (a.get('relevance_score') or 0) >= relevance_threshold]
    if articles:
        st.write(f"**📊 Filtro por relevancia:** {len(filtered)}/{len(articles)} ({len(filtered)/len(articles)*100:.1f}%)")
    return filtered


# ============================================================================
# TABLA GRADE
# ============================================================================

def add_grade_table_to_doc(doc, articles):
    articles = deduplicate_articles(articles)
    top_articles = sorted(articles, key=lambda x: x.get('evidence_score', 0), reverse=True)[:10]
    
    doc.add_heading('📊 Tabla de Evidencia GRADE (top 10 estudios)', level=2)
    
    table = doc.add_table(rows=1, cols=7)
    table.style = 'Table Grid'
    
    headers = ['Estudio', 'Diseño', 'N', 'EF (IC95%)', 'Calidad', 'Relación', 'Score']
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = header
        cell.paragraphs[0].runs[0].bold = True
    
    for art in top_articles:
        row = table.add_row().cells
        authors = art.get('authors', 'Autor')[:30]
        year = art.get('pubdate', '')[:4]
        row[0].text = f"{authors} {year}"
        row[1].text = art.get('study_design', 'ND')[:15]
        n = art.get('sample_size', 'N/A')
        row[2].text = str(n) if n else 'N/A'
        
        effect_type = art.get('effect_type', '')
        effect_val = art.get('effect_value', 0)
        if effect_val and effect_val > 0:
            ci_lower = art.get('effect_ci_lower', 0)
            ci_upper = art.get('effect_ci_upper', 0)
            row[3].text = f"{effect_type}={effect_val:.2f} ({ci_lower:.2f}-{ci_upper:.2f})"
        else:
            row[3].text = 'NR'
        
        meth_score = art.get('methodological_score', 0) or 0
        if meth_score >= 70:
            quality = f"🟢 Alta ({meth_score:.0f})"
        elif meth_score >= 50:
            quality = f"🟡 Moderada ({meth_score:.0f})"
        else:
            quality = f"🟠 Baja ({meth_score:.0f})"
        row[4].text = quality
        
        relation_map = {
            'confirms': '✅ Confirma',
            'nuances': '🔍 Matiza', 
            'extends': '🚀 Extiende',
            'contradicts': '⚠️ Contradice'
        }
        relation = art.get('hypothesis_relation', 'unrelated')
        row[5].text = relation_map.get(relation, '📚 Otro')
        row[6].text = f"{art.get('evidence_score', 0):.0f}/100"
    
    doc.add_paragraph()


# ============================================================================
# GENERACIÓN DE FLAVORS
# ============================================================================

def generate_insight_for_relation(relation, articles, hypothesis):
    if not articles:
        return ""
    
    articles = deduplicate_articles(articles)
    n = len(articles)
    best = articles[0]
    best_evidence = best.get('evidence_score') or 0
    best_design = best.get('study_design', 'Estudio')
    
    effect_str = ""
    effect_val = best.get('effect_value')
    if effect_val and effect_val > 0:
        effect_str = f" (HR={effect_val})"
    
    insights = {
        'confirms': f"**{n} estudios** confirman la hipótesis. El estudio de mayor nivel de evidencia es un {best_design}{effect_str} con puntaje de evidencia {best_evidence}/100.",
        'nuances': f"**{n} estudios** matizan la hipótesis, añadiendo condiciones específicas.",
        'extends': f"**{n} estudios** extienden el alcance de la hipótesis.",
        'contradicts': f"**{n} estudios** presentan evidencia que contradice parcialmente la hipótesis.",
        'unrelated': f"**{n} estudios** no abordan directamente la hipótesis."
    }
    return insights.get(relation, f"{n} estudios analizados.")


def generate_flavors_by_thesis(articles, hypothesis):
    if not articles:
        return {}
    
    articles = deduplicate_articles(articles)
    
    relation_groups = {
        'confirms': [], 'nuances': [], 'extends': [], 
        'contradicts': [], 'unrelated': []
    }
    
    for article in articles:
        relation = article.get('hypothesis_relation', 'unrelated')
        if relation in relation_groups:
            relation_groups[relation].append(article)
        else:
            relation_groups['unrelated'].append(article)
    
    for group in relation_groups.values():
        group.sort(key=lambda x: x.get('evidence_score') or 0, reverse=True)
    
    flavors = {}
    relation_names = {
        'confirms': ('✅ Confirma la Hipótesis', 'Artículos que validan directamente la hipótesis'),
        'nuances': ('🔍 Matiza la Hipótesis', 'Artículos que añaden condiciones o matices'),
        'extends': ('🚀 Extiende la Hipótesis', 'Artículos que amplían el alcance'),
        'contradicts': ('⚠️ Contradice la Hipótesis', 'Artículos con evidencia contraria'),
        'unrelated': ('📚 No Relacionados', 'Artículos que no abordan directamente la hipótesis')
    }
    
    for relation, articles_group in relation_groups.items():
        if not articles_group:
            continue
        
        name, description = relation_names.get(relation, ('Otros', ''))
        ev_scores = [a.get('evidence_score') or 0 for a in articles_group]
        avg_evidence = sum(ev_scores) / len(ev_scores) if ev_scores else 0
        has_effect = sum(1 for a in articles_group if a.get('effect_value') and a.get('effect_value') > 0)
        
        insight = generate_insight_for_relation(relation, articles_group, hypothesis)
        
        unique_repr = []
        seen_repr = set()
        for art in articles_group[:5]:
            pmid = art.get('pmid', '')
            if pmid and pmid not in seen_repr:
                seen_repr.add(pmid)
                unique_repr.append(art)
        
        flavors[relation] = {
            'name': name,
            'description': description,
            'articles': articles_group,
            'n_articles': len(articles_group),
            'avg_evidence_score': avg_evidence,
            'has_effect_size': has_effect,
            'insight': insight,
            'representative_articles': unique_repr
        }
    
    return flavors


def generate_traditional_clusters(articles, query_terms, hypothesis_terms):
    if len(articles) < 3:
        return []
    
    articles = deduplicate_articles(articles)
    
    try:
        texts = [f"{a.get('title', '')} {' '.join(a.get('all_outcomes', []))}" for a in articles]
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        svd = TruncatedSVD(n_components=min(50, tfidf_matrix.shape[1] - 1), random_state=42)
        embeddings_reduced = svd.fit_transform(tfidf_matrix)
        
        n_clusters = min(max(2, len(articles) // 10), 4)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings_reduced)
        
        flavors = []
        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
            if len(cluster_indices) < 2:
                continue
            cluster_articles = [articles[i] for i in cluster_indices]
            cluster_articles.sort(key=lambda x: x.get('evidence_score') or 0, reverse=True)
            
            all_outcomes = []
            for a in cluster_articles:
                all_outcomes.extend(a.get('all_outcomes', []))
            outcome_counts = Counter(all_outcomes)
            top_outcomes = [o for o, _ in outcome_counts.most_common(2)]
            name = f"Cluster sobre {', '.join(top_outcomes)}" if top_outcomes else f"Cluster {cluster_id + 1}"
            
            ev_scores = [a.get('evidence_score') or 0 for a in cluster_articles]
            avg_evidence = sum(ev_scores) / len(ev_scores) if ev_scores else 0
            
            unique_repr = []
            seen_repr = set()
            for art in cluster_articles[:5]:
                pmid = art.get('pmid', '')
                if pmid and pmid not in seen_repr:
                    seen_repr.add(pmid)
                    unique_repr.append(art)
            
            flavors.append({
                'type': 'traditional_cluster',
                'id': f"cluster_{cluster_id}",
                'name': name,
                'articles': cluster_articles,
                'n_articles': len(cluster_articles),
                'representative_articles': unique_repr,
                'avg_evidence': avg_evidence
            })
        return flavors
    except Exception as e:
        st.warning(f"Clustering falló: {e}")
        return []


def generate_all_flavors_human_style(articles, hypothesis, query_terms, hypothesis_terms, method="both"):
    if not articles:
        return {}
    
    articles = deduplicate_articles(articles)
    flavors = {}
    
    if method in ["thesis", "both"]:
        flavors['by_thesis'] = generate_flavors_by_thesis(articles, hypothesis)
    
    if method in ["clustering", "both"]:
        flavors['by_cluster'] = generate_traditional_clusters(articles, query_terms, hypothesis_terms)
    
    return flavors


def generate_flavor_summary_human_style(flavor_data, flavor_type, hypothesis):
    articles = flavor_data.get('articles', [])
    if not articles:
        return "No hay artículos.", []
    
    articles = deduplicate_articles(articles)
    n = len(articles)
    insight = flavor_data.get('insight', '')
    name = flavor_data.get('name', 'Flavor')
    
    high_evidence = [a for a in articles if (a.get('evidence_score') or 0) >= 80]
    
    if flavor_type == 'by_thesis':
        summary = f"### {name}\n\n{insight}\n\n"
        if high_evidence:
            summary += f"**📊 Evidencia de alta calidad:** {len(high_evidence)} estudios con nivel de evidencia ≥80/100.\n\n"
        
        best = articles[0] if articles else None
        if best:
            summary += f"**⭐ Estudio destacado:** {best.get('authors', 'Autor')[:80]} ({best.get('pubdate', 's.f.')[:4]}). "
            effect_val = best.get('effect_value')
            if effect_val and effect_val > 0:
                summary += f"Reporta {best.get('effect_type', 'efecto')}={effect_val}. "
            summary += f"Nivel de evidencia: {best.get('evidence_score') or 0}/100.\n\n"
    else:
        summary = f"### {flavor_data.get('name', 'Cluster')}\n\n"
        summary += f"Este grupo agrupa {n} artículos con temáticas similares. "
        summary += f"Calidad media de evidencia: {flavor_data.get('avg_evidence', 0):.0f}/100.\n\n"
    
    citations = []
    seen_pmids = set()
    for i, article in enumerate(articles[:10], 1):
        pmid = str(article.get('pmid', ''))
        if pmid and pmid in seen_pmids:
            continue
        if pmid:
            seen_pmids.add(pmid)
        
        authors = str(article.get('authors', 'Autor'))[:50]
        title = str(article.get('title', 'Sin título'))[:80]
        journal = str(article.get('journal', 'Revista'))[:30]
        year = str(article.get('pubdate', 's.f.'))[:4]
        citation = f"{i}. {authors}. {title}. {journal}. {year}"
        if pmid and pmid != 'N/A':
            citation += f"; PMID: {pmid}"
        citations.append(citation)
    
    return summary, citations


# ============================================================================
# DOCUMENTO COMPLETO
# ============================================================================

def create_document_with_flavors_human_style(flavors, hypothesis, query, total_articles, relevance_threshold, consensus):
    doc = Document()
    
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
    
    title = doc.add_heading('Análisis de Evidencia por Flavors v8.4', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph(f'**Estrategia de búsqueda:** {str(query)[:200]}...')
    doc.add_paragraph(f'**Hipótesis:** "{str(hypothesis)[:200]}..."')
    doc.add_paragraph(f'**Artículos analizados:** {total_articles}')
    doc.add_paragraph(f'**Threshold de relevancia:** {relevance_threshold}')
    doc.add_paragraph(f'**Fecha de generación:** {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    doc.add_paragraph()
    
    doc.add_page_break()
    
    # RESUMEN EJECUTIVO
    exec_title = doc.add_heading('📋 RESUMEN EJECUTIVO Y ANÁLISIS DE CONSENSO', level=1)
    exec_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    all_articles = []
    for relation in ['confirms', 'nuances', 'extends', 'contradicts']:
        if relation in flavors.get('by_thesis', {}):
            all_articles.extend(flavors['by_thesis'][relation].get('articles', []))
    
    exec_summary = ConsensusAnalyzer.generate_executive_summary(all_articles, hypothesis, query)
    
    for line in exec_summary.split('\n'):
        if line.strip():
            if line.startswith('╔') or line.startswith('╚') or line.startswith('║'):
                p = doc.add_paragraph()
                p.add_run(line).font.size = Pt(8)
            elif line.startswith('🔬') or line.startswith('📊') or line.startswith('⭐') or line.startswith('📅') or line.startswith('💡'):
                p = doc.add_paragraph()
                run = p.add_run(line)
                run.bold = True
                run.font.size = Pt(11)
            elif line.startswith('   •'):
                doc.add_paragraph(line, style='List Bullet')
            else:
                doc.add_paragraph(line)
    
    doc.add_paragraph()
    
    if MATPLOTLIB_AVAILABLE:
        try:
            consensus_fig = plot_consensus_gauge(consensus)
            if consensus_fig:
                doc.add_picture(consensus_fig, width=Inches(5))
        except:
            pass
    
    doc.add_page_break()
    
    # ALERTAS DE CALIDAD
    doc.add_heading('🚦 Alertas de Calidad Metodológica', level=1)
    quality_alerts = QualityAlertSystem.generate_alerts(all_articles)
    doc.add_paragraph(f"**Nivel general:** {quality_alerts['overall_level']}")
    doc.add_paragraph(f"**{quality_alerts['overall_message']}**")
    doc.add_paragraph()
    
    if quality_alerts['alerts']['critical']:
        doc.add_heading('🔴 Alertas CRÍTICAS', level=2)
        for alert in quality_alerts['alerts']['critical']:
            doc.add_paragraph(f"• {alert['message']}", style='List Bullet')
            doc.add_paragraph(f"  → {alert['recommendation']}", style='List Bullet')
        doc.add_paragraph()
    
    if quality_alerts['alerts']['warning']:
        doc.add_heading('🟡 Advertencias', level=2)
        for alert in quality_alerts['alerts']['warning']:
            doc.add_paragraph(f"• {alert['message']}", style='List Bullet')
        doc.add_paragraph()
    
    if quality_alerts['alerts']['success']:
        doc.add_heading('🟢 Fortalezas metodológicas', level=2)
        for alert in quality_alerts['alerts']['success']:
            doc.add_paragraph(f"• {alert['message']}", style='List Bullet')
        doc.add_paragraph()
    
    doc.add_page_break()
    
    # CONTRADICCIONES
    doc.add_heading('⚠️ Detección de Contradicciones entre Estudios', level=1)
    contradictions = ContradictionDetector.find_contradictions(all_articles)
    contradiction_summary = ContradictionDetector.generate_contradiction_summary(contradictions)
    for line in contradiction_summary.split('\n'):
        if line.strip():
            doc.add_paragraph(line)
    doc.add_paragraph()
    
    if contradictions:
        doc.add_heading('Detalle:', level=2)
        for i, contra in enumerate(contradictions[:5], 1):
            conflict = contra['conflict']
            doc.add_paragraph(f"{i}. **{conflict['description']}**")
            doc.add_paragraph(f"   {conflict['detail']}", style='List Bullet')
    
    doc.add_page_break()
    
    # LÍNEA DE TIEMPO
    doc.add_heading('📅 Evolución Temporal del Consenso', level=1)
    temporal_data = TemporalConsensusAnalyzer.analyze_temporal_consensus(all_articles)
    temporal_summary = TemporalConsensusAnalyzer.generate_temporal_summary(temporal_data)
    for line in temporal_summary.split('\n'):
        if line.strip():
            doc.add_paragraph(line)
    
    temporal_plot = TemporalConsensusAnalyzer.create_temporal_plot(temporal_data)
    if temporal_plot:
        doc.add_picture(temporal_plot, width=Inches(6))
        doc.add_paragraph("Figura: Evolución del consenso científico por año.")
    else:
        doc.add_paragraph("No hay suficientes datos para gráfico temporal.")
    
    doc.add_page_break()
    
    # TABLA GRADE
    add_grade_table_to_doc(doc, all_articles)
    doc.add_page_break()
    
    # FLAVORS
    doc.add_heading('FLAVORS POR RELACIÓN CON LA HIPÓTESIS', level=1)
    doc.add_paragraph('Los siguientes flavors agrupan artículos según cómo se relacionan con la hipótesis.')
    doc.add_paragraph()
    
    thesis_flavors = flavors.get('by_thesis', {})
    for relation, flavor_data in thesis_flavors.items():
        if not flavor_data.get('articles'):
            continue
        
        summary, citations = generate_flavor_summary_human_style(flavor_data, 'by_thesis', hypothesis)
        doc.add_paragraph(summary)
        
        if citations:
            doc.add_paragraph('**Referencias clave:**', style='List Bullet')
            for citation in citations[:10]:
                doc.add_paragraph(citation, style='List Bullet')
        doc.add_paragraph()
    
    doc.add_page_break()
    
    # CLUSTERS
    if flavors.get('by_cluster'):
        doc.add_heading('CLUSTERS TEMÁTICOS', level=1)
        doc.add_paragraph('Clusters por similitud temática.')
        doc.add_paragraph()
        
        for cluster in flavors.get('by_cluster', []):
            if not cluster.get('articles'):
                continue
            
            summary, citations = generate_flavor_summary_human_style(cluster, 'by_cluster', hypothesis)
            doc.add_paragraph(summary)
            
            if citations:
                doc.add_paragraph('**Artículos en este cluster:**', style='List Bullet')
                for citation in citations[:10]:
                    doc.add_paragraph(citation, style='List Bullet')
            doc.add_paragraph()
        
        doc.add_page_break()
    
    # TABLA RESUMEN
    doc.add_heading('RESUMEN DE EVIDENCIA POR CATEGORÍA', level=1)
    table = doc.add_table(rows=1, cols=4)
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Relación con Hipótesis'
    hdr_cells[1].text = 'N° Artículos'
    hdr_cells[2].text = 'Score Evidencia'
    hdr_cells[3].text = 'Con tamaño del efecto'
    
    for relation, flavor_data in thesis_flavors.items():
        if not flavor_data.get('articles'):
            continue
        
        row_cells = table.add_row().cells
        row_cells[0].text = flavor_data.get('name', relation)
        row_cells[1].text = str(flavor_data.get('n_articles', 0))
        row_cells[2].text = f"{flavor_data.get('avg_evidence_score', 0):.0f}"
        row_cells[3].text = f"{flavor_data.get('has_effect_size', 0)} estudios"
    
    doc.add_paragraph()
    doc.add_paragraph('---')
    doc.add_paragraph(f'*Documento generado por PubMed AI Analyzer v8.4*')
    doc.add_paragraph(f'*Email configurado con: {get_email_config()["email_user"]}*')
    
    return doc


def export_articles_to_csv_enhanced(articles):
    if not articles:
        return None
    
    articles = deduplicate_articles(articles)
    data = []
    for article in articles:
        row = {
            'PMID': article.get('pmid', ''),
            'Título': article.get('title', ''),
            'Autores': article.get('authors', ''),
            'Revista': article.get('journal', ''),
            'Año': article.get('pubdate', '')[:4],
            'Diseño': article.get('study_design', ''),
            'Nivel evidencia': article.get('evidence_level', ''),
            'Score evidencia': article.get('evidence_score', 0),
            'Score metodológico': article.get('methodological_score', 0),
            'Tamaño muestral': article.get('sample_size', ''),
            'Efecto': article.get('effect_type', ''),
            'Valor efecto': article.get('effect_value', 0),
            'Relación hipótesis': article.get('hypothesis_relation', ''),
            'PICO_Población': article.get('pico_population', ''),
            'PICO_Intervención': article.get('pico_intervention', ''),
            'PICO_Outcomes': article.get('pico_outcomes', '')
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    return csv_buffer.getvalue().encode('utf-8-sig')


def display_flavors_preview_v8(flavors, consensus):
    st.markdown("---")
    st.markdown("## 📋 RESUMEN EJECUTIVO")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ Confirman", f"{consensus.get('confirms_percentage', 0):.0f}%")
    with col2:
        st.metric("⚠️ Contradicen", f"{consensus.get('contradicts_percentage', 0):.0f}%")
    with col3:
        st.metric("🎯 Consenso", consensus.get('consensus_level', 'ND'))
    with col4:
        st.metric("⭐ Alta Calidad", consensus.get('high_quality_confirms', 0))
    
    st.info(consensus.get('consensus_message', ''))
    
    if MATPLOTLIB_AVAILABLE:
        consensus_fig = plot_consensus_gauge(consensus)
        if consensus_fig:
            st.image(consensus_fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("## 🎨 Flavors Generados")
    
    thesis_flavors = flavors.get('by_thesis', {})
    if thesis_flavors:
        st.markdown("### 📚 Por relación con la hipótesis")
        for relation, flavor_data in thesis_flavors.items():
            if not flavor_data.get('articles'):
                continue
            with st.expander(f"{flavor_data.get('name', relation)} ({flavor_data.get('n_articles', 0)} artículos)", expanded=True):
                st.markdown(f"**{flavor_data.get('description', '')}**")
                st.markdown(f"📊 **Insight:** {flavor_data.get('insight', '')}")
                st.markdown(f"⭐ **Score evidencia promedio:** {flavor_data.get('avg_evidence_score', 0):.0f}/100")
                
                st.markdown("**📄 Artículos destacados:**")
                for i, art in enumerate(flavor_data.get('representative_articles', [])[:5], 1):
                    title = art.get('title', 'Sin título')[:80]
                    evidence = art.get('evidence_score') or 0
                    st.markdown(f"   {i}. {title}... (ev={evidence})")


def display_graphs_section(articles):
    if not MATPLOTLIB_AVAILABLE:
        st.info("📊 Instala matplotlib para gráficos")
        return
    
    articles = deduplicate_articles(articles)
    st.markdown("## 📊 Análisis Gráfico")
    
    # Gráfico temporal
    st.markdown("### 📅 Evolución temporal")
    temporal_data = TemporalConsensusAnalyzer.analyze_temporal_consensus(articles)
    temporal_plot = TemporalConsensusAnalyzer.create_temporal_plot(temporal_data)
    if temporal_plot:
        st.image(temporal_plot, use_container_width=True)
    else:
        st.info("No hay suficientes datos para gráfico temporal.")
    
    # Alertas
    st.markdown("### 🚦 Alertas de calidad")
    quality_alerts = QualityAlertSystem.generate_alerts(articles)
    st.markdown(f"**Nivel general:** {quality_alerts['overall_level']}")
    
    if quality_alerts['alerts']['critical']:
        with st.expander("🔴 Alertas CRÍTICAS", expanded=True):
            for alert in quality_alerts['alerts']['critical']:
                st.warning(f"**{alert['message']}**\n\n→ {alert['recommendation']}")
    
    if quality_alerts['alerts']['warning']:
        with st.expander("🟡 Advertencias", expanded=False):
            for alert in quality_alerts['alerts']['warning']:
                st.info(alert['message'])
    
    # Contradicciones
    st.markdown("### ⚠️ Contradicciones")
    contradictions = ContradictionDetector.find_contradictions(articles)
    if contradictions:
        st.warning(f"Se detectaron {len(contradictions)} contradicción(es)")
        for contra in contradictions[:3]:
            conflict = contra['conflict']
            st.markdown(f"**{conflict['description']}**")
            st.caption(conflict['detail'])
    else:
        st.success("✅ No se detectaron contradicciones")


def extract_key_terms_from_query(query):
    terms = set()
    mesh_pattern = r'"([^"]+)"\[Mesh\]'
    for term in re.findall(mesh_pattern, query, re.IGNORECASE):
        terms.add(term.lower())
    tiab_pattern = r'"([^"]+)"\[tiab\]'
    for term in re.findall(tiab_pattern, query, re.IGNORECASE):
        terms.add(term.lower())
    return list(terms)


def extract_key_terms_from_hypothesis(hypothesis):
    if not hypothesis:
        return []
    terms = set()
    words = hypothesis.lower().split()
    for i in range(len(words)-1):
        if len(words[i]) > 4 and len(words[i+1]) > 3:
            terms.add(f"{words[i]} {words[i+1]}")
    return list(terms)


def process_and_generate_flavors(query, hypothesis, threshold, user_email, session_manager=None, flavor_method="both"):
    start_time = time.time()
    
    query_terms = extract_key_terms_from_query(query)
    hypothesis_terms = extract_key_terms_from_hypothesis(hypothesis)
    st.info(f"📝 Términos: {len(query_terms)} búsqueda, {len(hypothesis_terms)} hipótesis")
    
    progress_placeholder = st.empty()
    
    def update_progress(current, total):
        progress_placeholder.progress(current / total, text=f"Procesando {current}/{total}...")
    
    with st.spinner("🔍 Buscando artículos..."):
        id_list, total = search_pubmed_complete(query.strip(), max_articles=3000)
        if not id_list:
            st.error("❌ No se encontraron artículos")
            return None, None, None, None, None
    
    all_articles = []
    BLOCK_SIZE = 500
    total_blocks = (len(id_list) + BLOCK_SIZE - 1) // BLOCK_SIZE
    blocks_to_process = min(total_blocks, 6)
    
    overall_progress = st.progress(0)
    article_count = 0
    
    for block_num in range(1, blocks_to_process + 1):
        start_idx = (block_num - 1) * BLOCK_SIZE
        end_idx = min(block_num * BLOCK_SIZE, len(id_list))
        block_ids = id_list[start_idx:end_idx]
        
        st.info(f"📦 Bloque {block_num}/{blocks_to_process}...")
        
        def block_progress(current, total):
            nonlocal article_count
            article_count += 1
            overall_progress.progress(article_count / len(id_list))
        
        block_articles = fetch_articles_details(
            block_ids, query_terms, hypothesis_terms, 
            hypothesis=hypothesis, block_number=block_num,
            progress_callback=block_progress
        )
        
        if block_articles:
            all_articles.extend(block_articles)
            if session_manager and session_manager.current_session_id:
                session_manager.save_articles_batch(block_articles, session_manager.current_session_id)
        
        time.sleep(2)
    
    progress_placeholder.empty()
    overall_progress.empty()
    
    if not all_articles:
        st.error("❌ No se procesaron artículos")
        return None, None, None, None, None
    
    st.success(f"✅ Procesados {len(all_articles)} artículos")
    
    with st.spinner("🧠 Calculando relevancia..."):
        all_articles = calculate_relevance_to_hypothesis(all_articles, hypothesis)
        filtered_articles = filter_articles_by_relevance(all_articles, threshold)
    
    filtered_articles = deduplicate_articles(filtered_articles)
    st.info(f"📊 Después de deduplicación: {len(filtered_articles)} artículos únicos")
    
    if len(filtered_articles) < 3:
        st.error(f"❌ Pocos artículos ({len(filtered_articles)}). Mínimo 3.")
        return None, None, None, None, None
    
    st.info(f"⏱️ Tiempo: {(time.time()-start_time)/60:.1f} minutos")
    
    consensus = ConsensusAnalyzer.calculate_consensus(filtered_articles)
    display_graphs_section(filtered_articles)
    
    with st.spinner(f"🎨 Generando flavors..."):
        flavors = generate_all_flavors_human_style(filtered_articles, hypothesis, query_terms, hypothesis_terms, method=flavor_method)
    
    return flavors, filtered_articles, query, hypothesis, consensus


# ============================================================================
# CLASE PARA ALMACENAMIENTO LOCAL
# ============================================================================

class LocalStorage:
    def __init__(self, login):
        self.login = login
        self.data_dir = f"./data_{login}"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def read_csv(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        return None
    
    def write_csv(self, filename, df):
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        return True
    
    def append_to_csv(self, filename, new_data):
        existing_df = self.read_csv(filename)
        if existing_df is not None and not existing_df.empty:
            df = pd.concat([existing_df, new_data], ignore_index=True)
        else:
            df = new_data
        return self.write_csv(filename, df)


class UserSessionManager:
    def __init__(self, login):
        self.login = login
        self.storage = LocalStorage(login)
        self.articles_file = f"{login}_articles_v84.csv"
        self.sessions_file = f"{login}_sessions_v84.csv"
        self.current_session_id = None
    
    def create_session(self, query: str, hypothesis: str, relevance_threshold: float, email: str) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        session_id = f"{timestamp}_{query_hash}"
        self.current_session_id = session_id
        
        session_data = pd.DataFrame([{
            'session_id': session_id,
            'login': self.login,
            'query': query[:500],
            'hypothesis': hypothesis[:500],
            'relevance_threshold': relevance_threshold,
            'total_found': 0,
            'total_processed': 0,
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'flavors_generated': False,
            'user_email': email
        }])
        
        self.storage.append_to_csv(self.sessions_file, session_data)
        return session_id
    
    def save_articles_batch(self, articles: List[Dict], session_id: str):
        if not articles:
            return
        records = []
        for article in articles:
            record = {
                'pmid': str(article.get('pmid', '')),
                'session_id': str(session_id),
                'login': str(self.login),
                'title': str(article.get('title', ''))[:1000],
                'authors': str(article.get('authors', ''))[:500],
                'journal': str(article.get('journal', ''))[:200],
                'abstract': str(article.get('abstract', ''))[:5000],
                'quality_score': float(article.get('quality_score', 0) or 0),
                'evidence_level': str(article.get('evidence_level', '')),
                'evidence_score': int(article.get('evidence_score', 0) or 0),
                'sample_size': int(article.get('sample_size', 0) or 0),
                'effect_type': str(article.get('effect_type', '')),
                'effect_value': float(article.get('effect_value', 0) or 0),
                'effect_ci_lower': float(article.get('effect_ci_lower', 0) or 0),
                'effect_ci_upper': float(article.get('effect_ci_upper', 0) or 0),
                'hypothesis_relation': str(article.get('hypothesis_relation', '')),
                'methodological_score': float(article.get('methodological_score', 0) or 0),
                'relevance_score': float(article.get('relevance_score', 0) or 0),
                'processed_date': datetime.now().isoformat(),
                'block_number': int(article.get('block_number', 0) or 0),
                'pico_population': str(article.get('pico_population', '')),
                'pico_intervention': str(article.get('pico_intervention', '')),
                'pico_outcomes': str(article.get('pico_outcomes', ''))
            }
            records.append(record)
        
        df_new = pd.DataFrame(records)
        existing_df = self.storage.read_csv(self.articles_file)
        if existing_df is not None and not existing_df.empty:
            df = pd.concat([existing_df, df_new], ignore_index=True)
            df = df.drop_duplicates(subset=['pmid', 'session_id'], keep='last')
        else:
            df = df_new
        
        self.storage.write_csv(self.articles_file, df)
        st.success(f"✅ Guardados {len(records)} artículos")
    
    def get_session_articles(self, session_id: str) -> pd.DataFrame:
        df = self.storage.read_csv(self.articles_file)
        if df is None or df.empty:
            return pd.DataFrame()
        return df[df['session_id'] == session_id]
    
    def update_session(self, session_id: str, status: str, total_processed: int = None):
        df = self.storage.read_csv(self.sessions_file)
        if df is None or df.empty:
            return
        df = df.copy()
        mask = df['session_id'] == session_id
        if total_processed is not None:
            df.loc[mask, 'total_processed'] = total_processed
        df.loc[mask, 'status'] = status
        self.storage.write_csv(self.sessions_file, df)
    
    def mark_flavors_generated(self, session_id: str):
        df = self.storage.read_csv(self.sessions_file)
        if df is None or df.empty:
            return
        df = df.copy()
        mask = df['session_id'] == session_id
        df.loc[mask, 'flavors_generated'] = True
        self.storage.write_csv(self.sessions_file, df)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.title("🧠 PubMed AI Analyzer v8.4")
    st.markdown("*Análisis de evidencia con ENVÍO POR EMAIL, Alertas de Calidad, Detección de Contradicciones y Línea de Tiempo*")
    
    if 'user_login' not in st.session_state:
        st.markdown("### 🔐 Identificación")
        col1, col2 = st.columns(2)
        with col1:
            login = st.text_input("Login:", placeholder="ejemplo: investigador")
        with col2:
            user_email = st.text_input("📧 Email:", placeholder="usuario@dominio.com")
        
        if st.button("✅ Continuar", type="primary"):
            if login.strip() and user_email.strip() and '@' in user_email:
                st.session_state.user_login = login.strip().lower()
                st.session_state.user_email = user_email.strip()
                st.rerun()
            else:
                st.warning("⚠️ Complete ambos campos")
        return
    
    st.sidebar.success(f"👤 Usuario: {st.session_state.user_login}")
    st.sidebar.info(f"📧 {st.session_state.user_email}")
    
    if st.sidebar.button("🔄 Cambiar usuario"):
        del st.session_state.user_login
        del st.session_state.user_email
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Configuración")
    
    flavor_method = st.sidebar.radio(
        "Método de flavors:",
        ["both", "thesis", "clustering"],
        format_func=lambda x: {
            "both": "Ambos métodos",
            "thesis": "Solo por hipótesis",
            "clustering": "Solo clustering"
        }[x],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📧 Configuración de Email")
    
    email_config = EmailSender.get_email_config_status()
    if email_config['configured']:
        st.sidebar.success(f"✅ Email configurado")
        st.sidebar.info(f"📤 Desde: {email_config['sender_email']}")
        if email_config['notification_email'] != "No configurado":
            st.sidebar.info(f"📋 Copia a: {email_config['notification_email']}")
        
        # Botón de prueba
        if st.sidebar.button("📧 Probar conexión SMTP"):
            with st.spinner("Probando..."):
                success, msg = EmailSender.test_connection()
                if success:
                    st.sidebar.success(msg)
                else:
                    st.sidebar.error(msg)
    else:
        st.sidebar.error("❌ Email NO configurado")
        st.sidebar.info("Crear archivo .streamlit/secrets.toml con:")
        st.sidebar.code("""
smtp_server = "smtp.gmail.com"
smtp_port = 587
email_user = "tu_email@gmail.com"
email_password = "tu_contraseña"
notification_email = "copia@dominio.com"
        """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📦 Exportación")
    st.sidebar.markdown("- DOCX (con envío email)")
    st.sidebar.markdown("- CSV (datos completos)")
    st.sidebar.markdown("- RIS / BibTeX")
    
    st.info("⚡ **v8.4:** 📧 ENVÍO POR EMAIL | 🚦 Alertas de calidad | ⚠️ Detección de contradicciones | 📅 Línea de tiempo | 📊 Tabla GRADE")
    st.markdown("---")
    
    default_query = '("myocardial infarction"[Mesh] OR "myocardial infarction"[tiab]) AND ("intramyocardial dissection"[tiab] OR "intramyocardial dissecting hematoma"[tiab])'
    default_hypothesis = "Intramyocardial dissections occurring as a complication of myocardial infarction follow predictable anatomical pathways along established tissue planes."
    
    query = st.text_area("**PubMed search strategy:**", value=default_query, height=100)
    hypothesis = st.text_area("**📌 Hipótesis:**", value=default_hypothesis, height=100)
    threshold = st.slider("**Umbral de relevancia:**", 0.0, 0.9, 0.35, 0.05)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        generate_button = st.button("🚀 GENERAR FLAVORS (v8.4)", type="primary", use_container_width=True)
    with col2:
        send_email = st.checkbox("📧 Enviar por email", value=True)
    with col3:
        st.markdown("*Máx. 3000 artículos*")
    
    if generate_button:
        if not query.strip() or not hypothesis.strip():
            st.warning("⚠️ Complete búsqueda e hipótesis")
        else:
            session_manager = UserSessionManager(st.session_state.user_login)
            session_id = session_manager.create_session(query, hypothesis, threshold, st.session_state.user_email)
            
            with st.spinner("Procesando..."):
                flavors, articles, q, h, consensus = process_and_generate_flavors(
                    query, hypothesis, threshold, st.session_state.user_email, 
                    session_manager, flavor_method
                )
                
                if flavors and articles:
                    session_manager.mark_flavors_generated(session_id)
                    display_flavors_preview_v8(flavors, consensus)
                    
                    doc = create_document_with_flavors_human_style(
                        flavors, h, q, len(articles), threshold, consensus
                    )
                    
                    docx_bytes = BytesIO()
                    doc.save(docx_bytes)
                    docx_bytes.seek(0)
                    
                    st.markdown("---")
                    st.markdown("## 📥 Descargas")
                    
                    tab1, tab2, tab3, tab4 = st.tabs(["📄 DOCX", "📊 CSV", "📚 RIS", "📖 BibTeX"])
                    
                    with tab1:
                        filename = f"flavors_v84_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
                        st.download_button(
                            "💾 DESCARGAR DOCX",
                            data=docx_bytes,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )
                        
                        if send_email:
                            with st.spinner("📧 Enviando por correo..."):
                                success, message = EmailSender.send_docx_by_email(
                                    recipient_email=st.session_state.user_email,
                                    docx_bytes=docx_bytes,
                                    filename=filename
                                )
                                if success:
                                    st.success(message)
                                else:
                                    st.error(message)
                    
                    with tab2:
                        csv_data = export_articles_to_csv_enhanced(articles)
                        if csv_data:
                            st.download_button(
                                "📊 DESCARGAR CSV",
                                data=csv_data,
                                file_name=f"articles_v84_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    
                    with tab3:
                        ris_data = ReferenceExporter.to_ris(articles)
                        st.download_button(
                            "📚 DESCARGAR RIS",
                            data=ris_data,
                            file_name=f"references_v84_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ris",
                            mime="application/x-research-info-systems",
                            use_container_width=True
                        )
                    
                    with tab4:
                        bib_data = ReferenceExporter.to_bibtex(articles)
                        st.download_button(
                            "📖 DESCARGAR BibTeX",
                            data=bib_data,
                            file_name=f"references_v84_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bib",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    st.success("✅ ¡Proceso completado!")
                else:
                    st.error("❌ No se pudieron generar flavors. Intente con umbral más bajo.")
    
    st.markdown("---")
    st.markdown("*PubMed AI Analyzer v8.4 | Envío por email | Alertas de Calidad | Detección de Contradicciones | Línea de Tiempo*")


if __name__ == "__main__":
    main()
