"""
Trainer - Entraînement de 5 modèles avec tracking MLflow complet.

Modèles :
  1. Baseline (DummyClassifier)
  2. Logistic Regression
  3. Random Forest
  4. XGBoost
  5. Gradient Boosting (sklearn)
"""

import json
import logging
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

from src.ml.baseline import BaselineModel, compute_classification_metrics

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
EXPERIMENT_NAME = "edtech_dropout_prediction"

MODELS_CONFIG = {
    "baseline_majority": {
        "type": "baseline",
        "params": {"strategy": "most_frequent"},
    },
    "logistic_regression": {
        "type": "sklearn",
        "class": LogisticRegression,
        "params": {"max_iter": 1000, "random_state": 42, "class_weight": "balanced"},
    },
    "random_forest_v1": {
        "type": "sklearn",
        "class": RandomForestClassifier,
        "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42, "class_weight": "balanced"},
    },
    "random_forest_v2": {
        "type": "sklearn",
        "class": RandomForestClassifier,
        "params": {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5, "random_state": 42, "class_weight": "balanced"},
    },
    "gradient_boosting": {
        "type": "sklearn",
        "class": GradientBoostingClassifier,
        "params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": 42},
    },
    "xgboost": {
        "type": "xgboost",
        "class": XGBClassifier,
        "params": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": "logloss",
            "verbosity": 0,
        },
    },
}


def _get_feature_importance(model, feature_names: list) -> dict | None:
    """Extrait l'importance des features si disponible."""
    if hasattr(model, "feature_importances_"):
        return {
            name: round(float(imp), 6)
            for name, imp in zip(feature_names, model.feature_importances_)
        }
    if hasattr(model, "coef_"):
        return {
            name: round(float(abs(coef)), 6)
            for name, coef in zip(feature_names, model.coef_[0])
        }
    return None


def train_single_model(
    name: str,
    config: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Entraîne un modèle et log tout dans MLflow."""
    logger.info("Entraînement : %s", name)

    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_name", name)
        mlflow.log_params(config["params"])
        mlflow.set_tag("model_type", config["type"])
        mlflow.set_tag("dataset", "student_dropout")

        t0 = time.time()

        if config["type"] == "baseline":
            model_obj = BaselineModel(strategy=config["params"]["strategy"])
            model_obj.fit(X_train, y_train)
            y_pred = model_obj.predict(X_test)
            y_proba = model_obj.predict_proba(X_test)
            sklearn_model = model_obj.model
        else:
            sklearn_model = config["class"](**config["params"])
            sklearn_model.fit(X_train, y_train)
            y_pred = sklearn_model.predict(X_test)
            if hasattr(sklearn_model, "predict_proba"):
                y_proba = sklearn_model.predict_proba(X_test)[:, 1]
            else:
                y_proba = None

        train_time = round(time.time() - t0, 3)

        metrics = compute_classification_metrics(y_test.values, y_pred, y_proba)
        metrics["train_time_s"] = train_time

        mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
        mlflow.log_metric("train_size", len(X_train))
        mlflow.log_metric("test_size", len(X_test))

        t_inf = time.time()
        for _ in range(50):
            sklearn_model.predict(X_test.iloc[:10])
        inference_ms = round((time.time() - t_inf) / 50 * 1000, 3)
        mlflow.log_metric("inference_time_ms", inference_ms)
        metrics["inference_time_ms"] = inference_ms

        feature_imp = _get_feature_importance(sklearn_model, list(X_train.columns))
        if feature_imp:
            imp_path = MODELS_DIR / f"{name}_feature_importance.json"
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            with open(imp_path, "w") as f:
                json.dump(feature_imp, f, indent=2)
            mlflow.log_artifact(str(imp_path))

        if config["type"] == "xgboost":
            mlflow.xgboost.log_model(sklearn_model, "model")
        else:
            mlflow.sklearn.log_model(sklearn_model, "model")

        run_id = mlflow.active_run().info.run_id
        logger.info(
            "%s → acc=%.4f f1=%.4f roc_auc=%s train=%.2fs",
            name,
            metrics["accuracy"],
            metrics["f1"],
            metrics.get("roc_auc"),
            train_time,
        )

    return {
        "name": name,
        "model": sklearn_model,
        "metrics": metrics,
        "run_id": run_id,
        "config": config,
        "feature_importance": feature_imp,
    }


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tracking_uri: str = "http://localhost:5000",
) -> list[dict]:
    """Entraîne tous les modèles configurés et retourne les résultats."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("=== Entraînement Jour 3 : %d modèles ===", len(MODELS_CONFIG))
    results = []

    for name, config in MODELS_CONFIG.items():
        try:
            result = train_single_model(name, config, X_train, y_train, X_test, y_test)
            results.append(result)
        except Exception as exc:
            logger.error("Erreur modèle '%s' : %s", name, exc)

    logger.info("=== Entraînement terminé : %d/%d modèles réussis ===", len(results), len(MODELS_CONFIG))
    return results
