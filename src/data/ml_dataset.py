"""
Préparation du dataset ML-ready.

- Sélection des features utiles
- Encodage des catégorielles résiduelles
- Standardisation des variables numériques
- Split train/test
- Sauvegarde dans data/features/
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURES_DIR = Path("data/features")

ML_NUMERIC_FEATURES = [
    "Age",
    "Travel_Time",
    "Study_Time",
    "Number_of_Failures",
    "Family_Relationship",
    "Free_Time",
    "Going_Out",
    "Weekend_Alcohol_Consumption",
    "Weekday_Alcohol_Consumption",
    "Health_Status",
    "Number_of_Absences",
    "Grade_1",
    "Grade_2",
    "Final_Grade",
    "Mother_Education",
    "Father_Education",
    "academic_performance_score",
    "grade_progression",
    "alcohol_risk_score",
    "social_risk_index",
    "family_support_score",
    "engagement_score",
    "parental_education_level",
    "absence_risk_flag",
    "failure_history_flag",
    "combined_risk_score",
]

BINARY_FEATURES = [
    "School_Support",
    "Family_Support",
    "Extra_Paid_Class",
    "Extra_Curricular_Activities",
    "Attended_Nursery",
    "Wants_Higher_Education",
    "Internet_Access",
    "In_Relationship",
]

TARGET = "Dropped_Out"

COLS_TO_SCALE = [
    "Age",
    "Number_of_Absences",
    "Grade_1",
    "Grade_2",
    "Final_Grade",
    "academic_performance_score",
    "grade_progression",
    "family_support_score",
    "engagement_score",
    "parental_education_level",
    "combined_risk_score",
]


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sélectionne les colonnes features et la cible."""
    available_num = [c for c in ML_NUMERIC_FEATURES if c in df.columns]
    available_bin = [c for c in BINARY_FEATURES if c in df.columns]

    one_hot_cols = [c for c in df.columns if any(
        c.startswith(prefix + "_") for prefix in [
            "School", "Gender", "Address", "Family_Size", "Parental_Status",
            "Mother_Job", "Father_Job", "Reason_for_Choosing_School", "Guardian",
        ]
    )]

    feature_cols = list(dict.fromkeys(available_num + available_bin + one_hot_cols))
    feature_cols = [c for c in feature_cols if c != TARGET]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning("Features absentes du DataFrame : %s", missing)
        feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    logger.info("Features sélectionnées : %d colonnes", len(feature_cols))
    logger.info("Target : %s (classes : %s)", TARGET, sorted(y.unique().tolist()))
    return X, y


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Standardise les colonnes continues (mean=0, std=1)."""
    cols_to_scale = [c for c in COLS_TO_SCALE if c in X_train.columns]
    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    logger.info("Standardisation appliquée sur %d colonnes", len(cols_to_scale))
    return X_train, X_test, scaler


def save_ml_datasets(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Path]:
    """Sauvegarde les splits train/test dans data/features/."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}

    train = X_train.copy()
    train[TARGET] = y_train.values
    test = X_test.copy()
    test[TARGET] = y_test.values

    paths["train"] = FEATURES_DIR / "train.parquet"
    paths["test"] = FEATURES_DIR / "test.parquet"
    paths["X_train"] = FEATURES_DIR / "X_train.parquet"
    paths["X_test"] = FEATURES_DIR / "X_test.parquet"
    paths["y_train"] = FEATURES_DIR / "y_train.parquet"
    paths["y_test"] = FEATURES_DIR / "y_test.parquet"

    train.to_parquet(paths["train"], index=False)
    test.to_parquet(paths["test"], index=False)
    X_train.to_parquet(paths["X_train"], index=False)
    X_test.to_parquet(paths["X_test"], index=False)
    y_train.to_frame().to_parquet(paths["y_train"], index=False)
    y_test.to_frame().to_parquet(paths["y_test"], index=False)

    logger.info(
        "Datasets ML sauvegardés - train: %d lignes, test: %d lignes",
        len(X_train),
        len(X_test),
    )
    return paths


def build_ml_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
    """Orchestration complète de la préparation ML."""
    logger.info("=== Préparation dataset ML : début ===")

    X, y = select_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        "Split train/test : %d/%d (stratifié sur target)",
        len(X_train),
        len(X_test),
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    paths = save_ml_datasets(X_train_scaled, X_test_scaled, y_train, y_test)

    result = {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": list(X_train_scaled.columns),
        "paths": {k: str(v) for k, v in paths.items()},
    }

    logger.info("=== Préparation dataset ML : terminé ===")
    logger.info("Features finales : %d", len(result["feature_names"]))
    return result
