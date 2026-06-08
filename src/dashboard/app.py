import os

import requests
import streamlit as st


st.set_page_config(
    page_title="EdTech Analytics",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 EdTech - Analytics Pédagogique")
st.subheader("Jour 1 - Validation infrastructure")

api_url = os.getenv("API_URL", "http://api:8000")

col1, col2, col3 = st.columns(3)

try:
    response = requests.get(f"{api_url}/health", timeout=5)
    health = response.json()
except Exception:
    health = {
        "api": "unavailable",
        "database": "unavailable",
        "mlflow": "unavailable",
    }

col1.metric("API FastAPI", health.get("api", "unknown"))
col2.metric("PostgreSQL", health.get("database", "unknown"))
col3.metric("MLflow", health.get("mlflow", "unknown"))

st.divider()
st.markdown("### Architecture cible")
st.markdown(
    """
- Ingestion LMS vers PostgreSQL
- Feature engineering avec Pandas
- Modèles ML suivis avec MLflow
- API FastAPI pour l'inférence
- Dashboard Streamlit multi-rôles
- Monitoring Prometheus/Grafana
"""
)

st.markdown("### Prochaine étape")
st.info("Jour 2 : ingestion, processing, EDA rapide et dataset ML-ready.")
