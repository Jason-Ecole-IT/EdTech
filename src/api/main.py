from datetime import datetime, timezone
import os
from typing import List

import psycopg2
import requests
from fastapi import FastAPI
from pydantic import ValidationError

from src.api.predict import (
    StudentFeatures,
    PredictionResponse,
    predict_single,
    predict_batch,
)


app = FastAPI(
    title="EdTech Learning Analytics API",
    description="API Jour 4 avec prediction depuis MLflow Registry",
    version="0.2.0",
)


def get_database_status() -> str:
    try:
        connection = psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB", "edtech"),
            user=os.getenv("POSTGRES_USER", "edtech_user"),
            password=os.getenv("POSTGRES_PASSWORD", "edtech_password"),
            host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            connect_timeout=3,
        )
        connection.close()
        return "connected"
    except Exception:
        return "unavailable"


def get_mlflow_status() -> str:
    try:
        response = requests.get(
            f"{os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')}/health",
            timeout=3,
        )
        return "connected" if response.ok else "unavailable"
    except Exception:
        return "unavailable"


@app.get("/")
def root() -> dict[str, str]:
    return {
        "project": "EdTech Analytics Pedagogique",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "api": "healthy",
        "database": get_database_status(),
        "mlflow": get_mlflow_status(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: StudentFeatures) -> PredictionResponse:
    """Predit le decrochage pour un etudiant."""
    try:
        return predict_single(features)
    except ValidationError as e:
        raise ValueError(f"Validation error: {e}")
    except Exception as e:
        raise RuntimeError(f"Prediction error: {e}")


@app.post("/batch_predict", response_model=List[PredictionResponse])
def batch_predict(features_list: List[StudentFeatures]) -> List[PredictionResponse]:
    """Predit le decrochage pour plusieurs etudiants."""
    if not features_list:
        raise ValueError("features_list cannot be empty")
    try:
        return predict_batch(features_list)
    except ValidationError as e:
        raise ValueError(f"Validation error: {e}")
    except Exception as e:
        raise RuntimeError(f"Prediction error: {e}")
