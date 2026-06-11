import os
import requests
import streamlit as st
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

st.set_page_config(page_title="EdTech Analytics", layout="wide", page_icon="🎓")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
API_URL    = os.getenv("API_URL", "http://127.0.0.1:8000")
mlflow.set_tracking_uri(MLFLOW_URI)

st.title("🎓 EdTech - Analytics Pedagogique")

if st.button("🔄 Rafraichir"):
    st.rerun()

# Services status
st.subheader("Services")
try:
    r = requests.get(f"{API_URL}/health", timeout=3)
    h = r.json()
    st.success(f"API accessible : {API_URL}")
except Exception as e:
    h = {"api": "unavailable", "database": "unavailable", "mlflow": "unavailable"}
    st.error(f"API inaccessible : {e}")

c1, c2, c3 = st.columns(3)
c1.metric("API FastAPI", h.get("api", "?"))
c2.metric("PostgreSQL",  h.get("database", "?"))
c3.metric("MLflow",      h.get("mlflow", "?"))

st.divider()

# MLflow experiments
st.subheader("📊 MLflow - Comparaison des modeles")
try:
    client = MlflowClient(MLFLOW_URI)
    exp = client.get_experiment_by_name("edtech_dropout_prediction")
    if exp is None:
        st.warning("Experiment MLflow introuvable. Lancez: python -m src.ml.pipeline")
    else:
        runs = client.search_runs(exp.experiment_id, order_by=["metrics.f1 DESC"])
        rows = []
        seen = set()
        for run in runs:
            name = run.data.tags.get("model_name", run.info.run_name)
            if name in seen:
                continue
            seen.add(name)
            rows.append({
                "Modele":    name,
                "F1":        round(run.data.metrics.get("f1", 0), 4),
                "ROC-AUC":   round(run.data.metrics.get("roc_auc", 0), 4),
                "Accuracy":  round(run.data.metrics.get("accuracy", 0), 4),
                "Precision": round(run.data.metrics.get("precision", 0), 4),
                "Recall":    round(run.data.metrics.get("recall", 0), 4),
                "Train(s)":  round(run.data.metrics.get("train_time_s", 0), 3),
                "Infer(ms)": round(run.data.metrics.get("inference_time_ms", 0), 2),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        best = df.iloc[0]
        st.success(
            f"🏆 Meilleur modele : **{best['Modele']}** | "
            f"F1={best['F1']} | ROC-AUC={best['ROC-AUC']} | Accuracy={best['Accuracy']}"
        )

        st.markdown("#### Metriques par modele")
        chart_df = df.set_index("Modele")[["F1", "ROC-AUC", "Accuracy"]]
        st.bar_chart(chart_df)

except Exception as e:
    st.error(f"MLflow inaccessible : {e}")
    st.info("Lancez MLflow : mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db")

st.divider()

# Registry
st.subheader("📦 MLflow Registry")
try:
    client2 = MlflowClient(MLFLOW_URI)
    versions = client2.get_latest_versions("edtech_dropout_classifier")
    for v in versions:
        st.success(
            f"Modele : **edtech_dropout_classifier** | "
            f"Version : {v.version} | Stage : {v.current_stage}"
        )
except Exception as e:
    st.warning(f"Registry indisponible : {e}")

st.divider()

# Resultats Jour 2
st.subheader("📋 Resultats Jour 2 - Data Pipeline")
st.markdown("- 649 etudiants charges et nettoyes")
st.markdown("- 10 features engineering creees")
st.markdown("- 9/9 tests qualite passes")
st.markdown("- Dataset ML-ready : 519 train / 130 test | Taux decrochage : 15.4%")
st.info("Jour 4 : API FastAPI avec endpoint prediction depuis MLflow Registry")
