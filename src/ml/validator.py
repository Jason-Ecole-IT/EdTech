"""
Validation du modèle sélectionné avant production.

Tests : prédictions valides, reproductibilité, edge cases,
        sensibilité aux features, performance par classe.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


class ModelValidator:
    """Tests de validation complets du modèle ML."""

    def __init__(self, model, X_test: pd.DataFrame, y_test: pd.Series):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test.values if hasattr(y_test, "values") else np.array(y_test)
        self.results: list[dict] = []

    def _record(self, name: str, passed: bool, details: str = "") -> None:
        self.results.append({"test": name, "passed": passed, "details": details})
        symbol = "✓" if passed else "✗"
        level = logger.info if passed else logger.warning
        level("%s %s%s", symbol, name, f" - {details}" if details else "")

    def test_predictions_valid(self) -> bool:
        """Vérifie absence de NaN/Inf dans les prédictions."""
        y_pred = self.model.predict(self.X_test)
        has_nan = bool(np.any(np.isnan(y_pred.astype(float))))
        has_inf = bool(np.any(np.isinf(y_pred.astype(float))))
        passed = not has_nan and not has_inf
        self._record("Prédictions sans NaN/Inf", passed,
                     f"nan={has_nan}, inf={has_inf}")
        return passed

    def test_reproducibility(self) -> bool:
        """Vérifie que deux appels donnent les mêmes résultats."""
        y1 = self.model.predict(self.X_test)
        y2 = self.model.predict(self.X_test)
        passed = bool(np.array_equal(y1, y2))
        self._record("Reproductibilité des prédictions", passed)
        return passed

    def test_output_classes(self) -> bool:
        """Vérifie que les prédictions sont dans {0, 1}."""
        y_pred = self.model.predict(self.X_test)
        valid_classes = set(np.unique(y_pred)).issubset({0, 1})
        self._record("Classes de sortie valides {0,1}", valid_classes,
                     f"classes trouvées : {np.unique(y_pred).tolist()}")
        return valid_classes

    def test_single_sample(self) -> bool:
        """Vérifie la prédiction sur un seul échantillon."""
        try:
            single = self.X_test.iloc[:1]
            pred = self.model.predict(single)
            passed = len(pred) == 1
            self._record("Prédiction échantillon unique", passed)
        except Exception as e:
            self._record("Prédiction échantillon unique", False, str(e))
            passed = False
        return passed

    def test_batch_prediction(self) -> bool:
        """Vérifie que la taille de sortie correspond à l'entrée."""
        y_pred = self.model.predict(self.X_test)
        passed = len(y_pred) == len(self.X_test)
        self._record("Taille sortie = taille entrée", passed,
                     f"{len(y_pred)} vs {len(self.X_test)}")
        return passed

    def test_proba_range(self) -> bool:
        """Vérifie que les probabilités sont dans [0, 1]."""
        if not hasattr(self.model, "predict_proba"):
            self._record("Probabilités dans [0,1]", True, "predict_proba non disponible (skip)")
            return True
        try:
            proba = self.model.predict_proba(self.X_test)
            in_range = bool(np.all(proba >= 0) and np.all(proba <= 1))
            sums_to_one = bool(np.allclose(proba.sum(axis=1), 1.0, atol=1e-5))
            passed = in_range and sums_to_one
            self._record("Probabilités dans [0,1] et somme=1", passed)
        except Exception as e:
            self._record("Probabilités dans [0,1]", False, str(e))
            passed = False
        return passed

    def test_feature_sensitivity(self) -> bool:
        """Vérifie que le modèle réagit aux perturbations des features."""
        X_sample = self.X_test.iloc[:50].copy()
        baseline_pred = self.model.predict(X_sample)

        numeric_cols = X_sample.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            self._record("Sensibilité aux features", True, "Pas de colonnes numériques (skip)")
            return True

        X_perturbed = X_sample.copy()
        X_perturbed[numeric_cols[0]] = X_perturbed[numeric_cols[0]] + X_perturbed[numeric_cols[0]].std() * 2

        perturbed_pred = self.model.predict(X_perturbed)
        diff = int(np.sum(baseline_pred != perturbed_pred))
        passed = diff > 0
        self._record("Sensibilité aux features", passed,
                     f"{diff}/50 prédictions changées après perturbation")
        return passed

    def test_class_performance(self) -> dict:
        """Rapport de classification complet par classe."""
        y_pred = self.model.predict(self.X_test)
        report = classification_report(self.y_test, y_pred,
                                       target_names=["Actif (0)", "Décroché (1)"],
                                       output_dict=True)
        cm = confusion_matrix(self.y_test, y_pred)
        logger.info("Matrice de confusion :\n%s", cm)
        logger.info("Rapport par classe :\n%s",
                    classification_report(self.y_test, y_pred,
                                          target_names=["Actif (0)", "Décroché (1)"]))
        return {"classification_report": report, "confusion_matrix": cm.tolist()}

    def run_all_tests(self) -> dict:
        """Exécute tous les tests et retourne le rapport."""
        logger.info("=== Validation modèle : début ===")
        self.test_predictions_valid()
        self.test_reproducibility()
        self.test_output_classes()
        self.test_single_sample()
        self.test_batch_prediction()
        self.test_proba_range()
        self.test_feature_sensitivity()
        class_perf = self.test_class_performance()

        n_passed = sum(1 for r in self.results if r["passed"])
        n_total = len(self.results)
        all_passed = n_passed == n_total

        logger.info("=== Validation : %d/%d tests réussis ===", n_passed, n_total)

        return {
            "tests": self.results,
            "n_passed": n_passed,
            "n_total": n_total,
            "all_passed": all_passed,
            "class_performance": class_perf,
        }
