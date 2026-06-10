"""
Pipeline Jour 3 - Orchestration complète ML.

Étapes :
  1. Chargement des datasets ML-ready (Jour 2)
  2. Entraînement 6 modèles + MLflow tracking
  3. Comparaison et sélection du meilleur
  4. Validation du modèle sélectionné
  5. Enregistrement MLflow Registry (Production)
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/processed/pipeline_ml.log", mode="w", encoding="utf-8"),
    ],
)

logger = logging.getLogger(__name__)

import pandas as pd

from src.ml.comparison import (
    build_comparison_table,
    compute_composite_score,
    save_comparison_report,
    select_best_model,
)
from src.ml.registry import register_best_model
from src.ml.trainer import train_all_models
from src.ml.validator import ModelValidator

FEATURES_DIR = Path("data/features")
MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")

MLFLOW_URI = "http://localhost:5000"


def load_ml_datasets() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Charge les datasets ML-ready produits au Jour 2."""
    X_train = pd.read_parquet(FEATURES_DIR / "X_train.parquet")
    X_test = pd.read_parquet(FEATURES_DIR / "X_test.parquet")
    y_train = pd.read_parquet(FEATURES_DIR / "y_train.parquet").squeeze()
    y_test = pd.read_parquet(FEATURES_DIR / "y_test.parquet").squeeze()

    logger.info(
        "Datasets chargés : X_train=%s X_test=%s",
        X_train.shape,
        X_test.shape,
    )
    return X_train, y_train, X_test, y_test


def run_ml_pipeline() -> dict:
    """Exécute le pipeline ML complet Jour 3."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    logger.info("━" * 60)
    logger.info("ÉTAPE 1/5 : Chargement des datasets ML-ready")
    logger.info("━" * 60)
    X_train, y_train, X_test, y_test = load_ml_datasets()
    results["dataset"] = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": X_train.shape[1],
        "dropout_rate_train": round(float(y_train.mean()) * 100, 2),
    }

    logger.info("━" * 60)
    logger.info("ÉTAPE 2/5 : Entraînement des modèles + MLflow tracking")
    logger.info("━" * 60)
    model_results = train_all_models(X_train, y_train, X_test, y_test, tracking_uri=MLFLOW_URI)
    results["models_trained"] = len(model_results)

    logger.info("━" * 60)
    logger.info("ÉTAPE 3/5 : Comparaison et sélection du meilleur modèle")
    logger.info("━" * 60)
    selection = select_best_model(model_results)
    best = selection["best_result"]

    df_comp = build_comparison_table(model_results)
    df_comp = compute_composite_score(df_comp)
    save_comparison_report(df_comp, selection["justification"])

    results["best_model"] = {
        "name": best["name"],
        "metrics": best["metrics"],
        "run_id": best["run_id"],
        "justification": selection["justification"],
    }

    logger.info("━" * 60)
    logger.info("ÉTAPE 4/5 : Validation du modèle sélectionné")
    logger.info("━" * 60)
    validator = ModelValidator(best["model"], X_test, y_test)
    validation = validator.run_all_tests()
    results["validation"] = {
        "n_passed": validation["n_passed"],
        "n_total": validation["n_total"],
        "all_passed": validation["all_passed"],
    }

    val_path = PROCESSED_DIR / "model_validation_report.json"
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2, default=str)
    logger.info("Rapport validation sauvegardé : %s", val_path)

    logger.info("━" * 60)
    logger.info("ÉTAPE 5/5 : Enregistrement MLflow Registry")
    logger.info("━" * 60)
    registry_info = register_best_model(
        run_id=best["run_id"],
        model_name=best["name"],
        metrics=best["metrics"],
        justification=selection["justification"],
        tracking_uri=MLFLOW_URI,
    )
    results["registry"] = registry_info

    logger.info("━" * 60)
    logger.info("PIPELINE JOUR 3 TERMINÉ")
    logger.info("━" * 60)
    logger.info("Modèles entraînés   : %d", results["models_trained"])
    logger.info("Meilleur modèle     : %s", results["best_model"]["name"])
    logger.info("F1-score            : %.4f", results["best_model"]["metrics"].get("f1", 0))
    logger.info("ROC-AUC             : %.4f", results["best_model"]["metrics"].get("roc_auc") or 0)
    logger.info("Accuracy            : %.4f", results["best_model"]["metrics"].get("accuracy", 0))
    logger.info("Validation          : %d/%d tests réussis",
                results["validation"]["n_passed"], results["validation"]["n_total"])
    logger.info("Registry            : %s v%s (Production)",
                registry_info["registry_name"], registry_info["version"])
    logger.info("━" * 60)

    summary_path = PROCESSED_DIR / "jour3_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    run_ml_pipeline()
