"""
MLflow Model Registry - Enregistrement et promotion en Production.

Versionne le meilleur modèle et le promeut au stage Production
pour consommation par l'API (Jour 4).
"""

import json
import logging
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

MODEL_REGISTRY_NAME = "edtech_dropout_classifier"
MODELS_DIR = Path("models")


def register_best_model(
    run_id: str,
    model_name: str,
    metrics: dict,
    justification: str,
    tracking_uri: str = "http://localhost:5000",
) -> dict:
    """
    Enregistre le meilleur modèle dans le MLflow Model Registry
    et le promeut au stage Production.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    model_uri = f"runs:/{run_id}/model"
    logger.info("Enregistrement du modèle '%s' (run_id=%s)", MODEL_REGISTRY_NAME, run_id)

    mv = mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)
    version = mv.version
    logger.info("Modèle enregistré : version=%s", version)

    client.update_registered_model(
        name=MODEL_REGISTRY_NAME,
        description=(
            f"Modèle de prédiction du décrochage scolaire. "
            f"Entraîné sur Student Dropout dataset (649 étudiants). "
            f"Sélectionné : {model_name}. "
            f"Justification : {justification}"
        ),
    )

    client.update_model_version(
        name=MODEL_REGISTRY_NAME,
        version=version,
        description=(
            f"Version {version} - {model_name}\n"
            f"F1={metrics.get('f1', 'N/A')}, "
            f"ROC-AUC={metrics.get('roc_auc', 'N/A')}, "
            f"Accuracy={metrics.get('accuracy', 'N/A')}"
        ),
    )

    for stage_from in ["None", "Staging"]:
        try:
            client.transition_model_version_stage(
                name=MODEL_REGISTRY_NAME,
                version=version,
                stage="Production",
                archive_existing_versions=True,
            )
            break
        except Exception:
            pass

    logger.info(
        "Modèle '%s' v%s promu en Production ✓",
        MODEL_REGISTRY_NAME,
        version,
    )

    registry_info = {
        "registry_name": MODEL_REGISTRY_NAME,
        "version": version,
        "stage": "Production",
        "run_id": run_id,
        "model_name": model_name,
        "metrics": metrics,
        "model_uri": f"models:/{MODEL_REGISTRY_NAME}/Production",
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    info_path = MODELS_DIR / "registry_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(registry_info, f, indent=2, default=str)
    logger.info("Infos registry sauvegardées : %s", info_path)

    return registry_info


def get_production_model(tracking_uri: str = "http://localhost:5000"):
    """Charge le modèle en Production depuis le registry."""
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{MODEL_REGISTRY_NAME}/Production"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Modèle Production chargé : %s", model_uri)
        return model
    except Exception as exc:
        logger.error("Impossible de charger le modèle Production : %s", exc)
        raise
