"""
Baseline Models - Point de référence pour la comparaison.

Tout modèle avancé doit dépasser la baseline pour justifier sa complexité.
"""

import logging

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict:
    """Calcule les métriques de classification standard."""
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = round(roc_auc_score(y_true, y_proba), 4)
        except Exception:
            metrics["roc_auc"] = None
    return metrics


class BaselineModel:
    """
    Modèle baseline trivial - prédit la classe la plus fréquente.
    Représente le minimum absolu à dépasser.
    """

    def __init__(self, strategy: str = "most_frequent"):
        self.strategy = strategy
        self.model = DummyClassifier(strategy=strategy, random_state=42)
        self.metrics: dict = {}

    def fit(self, X_train, y_train) -> "BaselineModel":
        self.model.fit(X_train, y_train)
        logger.info("Baseline '%s' entraîné", self.strategy)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test) -> dict:
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        self.metrics = compute_classification_metrics(y_test, y_pred, y_proba)
        logger.info("Baseline metrics : %s", self.metrics)
        return self.metrics
