"""
Tests automatisés du pipeline ML Jour 3.

Couverture :
  - Baseline model
  - Métriques de classification
  - Comparaison de modèles
  - Validation du modèle
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from src.ml.baseline import BaselineModel, compute_classification_metrics
from src.ml.comparison import build_comparison_table, compute_composite_score, select_best_model
from src.ml.validator import ModelValidator


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def binary_dataset():
    """Dataset binaire simple pour les tests."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "feature_3": np.random.randint(0, 5, n).astype(float),
    })
    y = pd.Series((X["feature_1"] + X["feature_2"] > 0).astype(int), name="Dropped_Out")
    return X[:160], y[:160], X[160:], y[160:]


@pytest.fixture
def trained_rf(binary_dataset):
    X_train, y_train, X_test, y_test = binary_dataset
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


@pytest.fixture
def mock_results(binary_dataset):
    """Résultats simulés de plusieurs modèles."""
    X_train, y_train, X_test, y_test = binary_dataset
    results = []
    configs = [
        ("baseline", DummyClassifier(strategy="most_frequent"), {}),
        ("random_forest", RandomForestClassifier(n_estimators=10, random_state=42), {}),
    ]
    for name, model, _ in configs:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        metrics = compute_classification_metrics(y_test.values, y_pred, y_proba)
        results.append({
            "name": name,
            "model": model,
            "metrics": metrics,
            "run_id": f"fake_run_{name}",
            "config": {},
        })
    return results, X_test, y_test


# ──────────────────────────────────────────────
# Tests Baseline
# ──────────────────────────────────────────────

class TestBaseline:

    def test_baseline_fits_and_predicts(self, binary_dataset):
        X_train, y_train, X_test, y_test = binary_dataset
        baseline = BaselineModel(strategy="most_frequent")
        baseline.fit(X_train, y_train)
        y_pred = baseline.predict(X_test)
        assert len(y_pred) == len(X_test)

    def test_baseline_metrics_keys(self, binary_dataset):
        X_train, y_train, X_test, y_test = binary_dataset
        baseline = BaselineModel()
        baseline.fit(X_train, y_train)
        metrics = baseline.evaluate(X_test, y_test)
        for key in ["accuracy", "f1", "precision", "recall"]:
            assert key in metrics

    def test_baseline_accuracy_majority_class(self, binary_dataset):
        X_train, y_train, X_test, y_test = binary_dataset
        baseline = BaselineModel(strategy="most_frequent")
        baseline.fit(X_train, y_train)
        metrics = baseline.evaluate(X_test, y_test)
        majority_rate = max(y_test.mean(), 1 - y_test.mean())
        assert abs(metrics["accuracy"] - majority_rate) < 0.05


# ──────────────────────────────────────────────
# Tests Métriques
# ──────────────────────────────────────────────

class TestMetrics:

    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0

    def test_all_wrong_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0

    def test_roc_auc_with_proba(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = compute_classification_metrics(y_true, y_pred, y_proba)
        assert "roc_auc" in metrics
        assert metrics["roc_auc"] == 1.0

    def test_metrics_range(self, binary_dataset):
        X_train, y_train, X_test, y_test = binary_dataset
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = compute_classification_metrics(y_test.values, y_pred)
        for key in ["accuracy", "f1", "precision", "recall"]:
            assert 0.0 <= metrics[key] <= 1.0


# ──────────────────────────────────────────────
# Tests Comparaison
# ──────────────────────────────────────────────

class TestComparison:

    def test_comparison_table_has_all_models(self, mock_results):
        results, _, _ = mock_results
        df = build_comparison_table(results)
        assert len(df) == len(results)

    def test_comparison_sorted_by_f1(self, mock_results):
        results, _, _ = mock_results
        df = build_comparison_table(results)
        f1_vals = df["f1"].dropna().tolist()
        assert f1_vals == sorted(f1_vals, reverse=True)

    def test_composite_score_in_range(self, mock_results):
        results, _, _ = mock_results
        df = build_comparison_table(results)
        df = compute_composite_score(df)
        assert "composite_score" in df.columns
        assert df["composite_score"].between(0, 1).all()

    def test_select_best_returns_dict(self, mock_results):
        results, _, _ = mock_results
        selection = select_best_model(results)
        assert "best_result" in selection
        assert "justification" in selection
        assert "comparison_df" in selection

    def test_best_is_not_baseline(self, mock_results):
        results, _, _ = mock_results
        selection = select_best_model(results)
        assert selection["best_result"]["name"] != "baseline"


# ──────────────────────────────────────────────
# Tests Validator
# ──────────────────────────────────────────────

class TestModelValidator:

    def test_all_tests_pass_on_rf(self, trained_rf):
        model, X_test, y_test = trained_rf
        validator = ModelValidator(model, X_test, y_test)
        report = validator.run_all_tests()
        critical = [r for r in report["tests"] if not r["passed"]]
        assert len(critical) == 0, f"Tests échoués : {[r['test'] for r in critical]}"

    def test_report_structure(self, trained_rf):
        model, X_test, y_test = trained_rf
        validator = ModelValidator(model, X_test, y_test)
        report = validator.run_all_tests()
        assert "n_passed" in report
        assert "n_total" in report
        assert "all_passed" in report
        assert report["n_passed"] == report["n_total"]

    def test_predictions_valid(self, trained_rf):
        model, X_test, y_test = trained_rf
        validator = ModelValidator(model, X_test, y_test)
        assert validator.test_predictions_valid() is True

    def test_reproducibility(self, trained_rf):
        model, X_test, y_test = trained_rf
        validator = ModelValidator(model, X_test, y_test)
        assert validator.test_reproducibility() is True

    def test_output_classes(self, trained_rf):
        model, X_test, y_test = trained_rf
        validator = ModelValidator(model, X_test, y_test)
        assert validator.test_output_classes() is True
