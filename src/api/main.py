from datetime import datetime, timezone
import os

import psycopg2
import requests
from fastapi import FastAPI


app = FastAPI(
    title="EdTech Learning Analytics API",
    description="API Jour 1 pour valider la connectivité des services.",
    version="0.1.0",
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
