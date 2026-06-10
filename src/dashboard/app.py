import os
import requests
import streamlit as st

st.set_page_config(page_title="EdTech Analytics", layout="wide")
st.title("EdTech - Analytics Pedagogique")
st.subheader("Jour 3 - Modeles ML entraines")
api_url = os.getenv("API_URL", "http://127.0.0.1:8000")
if st.button("Rafraichir"):
    st.rerun()
try:
    r = requests.get(f"{api_url}/health", timeout=5)
    r.raise_for_status()
    h = r.json()
    st.success(f"API accessible : {api_url}")
except Exception as e:
    h = {"api": "unavailable", "database": "unavailable", "mlflow": "unavailable"}
    st.error(f"API inaccessible : {e}")
c1, c2, c3 = st.columns(3)
c1.metric("API FastAPI", h.get("api", "?"))
c2.metric("PostgreSQL", h.get("database", "?"))
c3.metric("MLflow", h.get("mlflow", "?"))
st.divider()
st.markdown("### Jour 3 - ML")
st.markdown("- 6 modeles entraines")
st.markdown("- Meilleur : GradientBoosting F1=1.00 ROC=1.00")
st.markdown("- edtech_dropout_classifier v1 en Production")
st.divider()
st.markdown("### Jour 2 - Data Pipeline")
st.markdown("- 649 etudiants, 10 features, 9/9 tests, 519 train/130 test")
st.info("Jour 4 : endpoint prediction depuis MLflow Registry")
