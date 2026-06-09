"""
Validation qualité des données - Tests automatisés et rapport.

Produit :
  - data/processed/quality_report.json : rapport de qualité complet
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")

EXPECTED_TYPES = {
    "Age": "numeric",
    "Travel_Time": "numeric",
    "Study_Time": "numeric",
    "Number_of_Failures": "numeric",
    "Family_Relationship": "numeric",
    "Free_Time": "numeric",
    "Going_Out": "numeric",
    "Weekend_Alcohol_Consumption": "numeric",
    "Weekday_Alcohol_Consumption": "numeric",
    "Health_Status": "numeric",
    "Number_of_Absences": "numeric",
    "Grade_1": "numeric",
    "Grade_2": "numeric",
    "Final_Grade": "numeric",
    "Dropped_Out": "numeric",
}

VALUE_RANGES = {
    "Age": (15, 22),
    "Travel_Time": (1, 4),
    "Study_Time": (1, 4),
    "Number_of_Failures": (0, 3),
    "Family_Relationship": (1, 5),
    "Free_Time": (1, 5),
    "Going_Out": (1, 5),
    "Weekend_Alcohol_Consumption": (1, 5),
    "Weekday_Alcohol_Consumption": (1, 5),
    "Health_Status": (1, 5),
    "Number_of_Absences": (0, 32),
    "Grade_1": (0, 20),
    "Grade_2": (0, 20),
    "Final_Grade": (0, 20),
    "Dropped_Out": (0, 1),
}

REQUIRED_COLUMNS = [
    "Age", "Study_Time", "Number_of_Failures", "Number_of_Absences",
    "Grade_1", "Grade_2", "Final_Grade", "Dropped_Out",
]


@dataclass
class ValidationResult:
    test_name: str
    passed: bool
    details: Any = None
    severity: str = "error"


@dataclass
class QualityReport:
    results: list[ValidationResult] = field(default_factory=list)

    def add(self, result: ValidationResult) -> None:
        self.results.append(result)
        symbol = "✓" if result.passed else "✗"
        level = logger.info if result.passed else (logger.warning if result.severity == "warning" else logger.error)
        level("%s [%s] %s", symbol, result.severity.upper(), result.test_name)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results if r.severity == "error")

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total_tests": len(self.results),
                "passed": self.n_passed,
                "failed": self.n_failed,
                "overall_passed": self.passed,
            },
            "tests": [
                {
                    "name": r.test_name,
                    "passed": r.passed,
                    "severity": r.severity,
                    "details": r.details,
                }
                for r in self.results
            ],
        }


def check_required_columns(df: pd.DataFrame, report: QualityReport) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report.add(ValidationResult(
        test_name="Colonnes requises présentes",
        passed=len(missing) == 0,
        details={"missing_columns": missing},
        severity="error",
    ))


def check_no_missing_values(df: pd.DataFrame, report: QualityReport) -> None:
    missing = df.isna().sum()
    missing_dict = {col: int(cnt) for col, cnt in missing.items() if cnt > 0}
    report.add(ValidationResult(
        test_name="Absence de valeurs manquantes",
        passed=len(missing_dict) == 0,
        details={"columns_with_missing": missing_dict},
        severity="error",
    ))


def check_no_duplicates(df: pd.DataFrame, report: QualityReport) -> None:
    n_dup = int(df.duplicated().sum())
    report.add(ValidationResult(
        test_name="Absence de doublons",
        passed=n_dup == 0,
        details={"duplicate_count": n_dup},
        severity="warning",
    ))


def check_column_types(df: pd.DataFrame, report: QualityReport) -> None:
    errors = {}
    for col, expected in EXPECTED_TYPES.items():
        if col not in df.columns:
            continue
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if expected == "numeric" and not is_numeric:
            errors[col] = {"expected": "numeric", "found": str(df[col].dtype)}
    report.add(ValidationResult(
        test_name="Types de colonnes corrects",
        passed=len(errors) == 0,
        details={"type_errors": errors},
        severity="error",
    ))


def check_value_ranges(df: pd.DataFrame, report: QualityReport) -> None:
    violations = {}
    for col, (lo, hi) in VALUE_RANGES.items():
        if col not in df.columns:
            continue
        out = int(((df[col] < lo) | (df[col] > hi)).sum())
        if out:
            violations[col] = {
                "expected_range": [lo, hi],
                "n_violations": out,
                "min_found": float(df[col].min()),
                "max_found": float(df[col].max()),
            }
    report.add(ValidationResult(
        test_name="Valeurs dans les plages attendues",
        passed=len(violations) == 0,
        details={"violations": violations},
        severity="warning",
    ))


def check_target_balance(df: pd.DataFrame, report: QualityReport) -> None:
    """Vérifie que la cible est présente et non vide d'une classe."""
    if "Dropped_Out" not in df.columns:
        return
    counts = df["Dropped_Out"].value_counts().to_dict()
    n_classes = len(counts)
    report.add(ValidationResult(
        test_name="Variable cible non dégénérée (2 classes)",
        passed=n_classes == 2,
        details={"class_counts": {int(k): int(v) for k, v in counts.items()}},
        severity="error",
    ))


def check_grade_consistency(df: pd.DataFrame, report: QualityReport) -> None:
    """Vérifie la cohérence : pas de note finale sans G1 et G2."""
    if not all(c in df.columns for c in ["Grade_1", "Grade_2", "Final_Grade"]):
        return
    incoherent = int(((df["Final_Grade"] > 0) & (df["Grade_1"] == 0) & (df["Grade_2"] == 0)).sum())
    report.add(ValidationResult(
        test_name="Cohérence des notes (G1, G2, Final)",
        passed=incoherent == 0,
        details={"incoherent_rows": incoherent},
        severity="warning",
    ))


def check_minimum_dataset_size(df: pd.DataFrame, report: QualityReport, min_rows: int = 100) -> None:
    report.add(ValidationResult(
        test_name=f"Dataset contient au moins {min_rows} lignes",
        passed=len(df) >= min_rows,
        details={"n_rows": len(df), "min_required": min_rows},
        severity="error",
    ))


def check_feature_columns_present(df: pd.DataFrame, report: QualityReport) -> None:
    expected_features = [
        "academic_performance_score",
        "grade_progression",
        "alcohol_risk_score",
        "social_risk_index",
        "engagement_score",
        "combined_risk_score",
        "absence_risk_flag",
        "failure_history_flag",
    ]
    missing = [f for f in expected_features if f not in df.columns]
    report.add(ValidationResult(
        test_name="Features engineered présentes",
        passed=len(missing) == 0,
        details={"missing_features": missing},
        severity="warning",
    ))


def run_validation(df: pd.DataFrame) -> QualityReport:
    """Exécute tous les tests de validation et retourne le rapport."""
    logger.info("=== Validation qualité : début ===")
    report = QualityReport()

    check_required_columns(df, report)
    check_no_missing_values(df, report)
    check_no_duplicates(df, report)
    check_column_types(df, report)
    check_value_ranges(df, report)
    check_target_balance(df, report)
    check_grade_consistency(df, report)
    check_minimum_dataset_size(df, report)
    check_feature_columns_present(df, report)

    logger.info(
        "=== Validation terminée : %d/%d tests réussis ===",
        report.n_passed,
        len(report.results),
    )
    return report


def save_quality_report(report: QualityReport) -> Path:
    """Sauvegarde le rapport qualité en JSON."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "quality_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("Rapport qualité sauvegardé : %s", out)
    return out
