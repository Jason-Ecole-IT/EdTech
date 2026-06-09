"""
Pipeline Jour 2 - Orchestration complète ETL → Features → EDA → Validation → ML-ready.

Usage :
    python -m src.data.pipeline
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/processed/pipeline.log", mode="w", encoding="utf-8"),
    ],
)

logger = logging.getLogger(__name__)

from src.data.eda import run_eda
from src.data.features import build_feature_set, save_features
from src.data.loader import clean, encode_categoricals, load_raw, save_processed
from src.data.ml_dataset import build_ml_dataset
from src.data.validation import run_validation, save_quality_report


def run_pipeline() -> dict:
    """
    Exécute le pipeline complet Jour 2.

    Retourne un dict avec les résultats de chaque étape.
    """
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/features").mkdir(parents=True, exist_ok=True)

    results = {}

    # --- Étape 1 : Chargement ---
    logger.info("━" * 60)
    logger.info("ÉTAPE 1/6 : Chargement des données brutes")
    logger.info("━" * 60)
    df_raw = load_raw()
    results["raw_shape"] = df_raw.shape

    # --- Étape 2 : Nettoyage ---
    logger.info("━" * 60)
    logger.info("ÉTAPE 2/6 : Nettoyage ETL")
    logger.info("━" * 60)
    df_clean = clean(df_raw)
    results["clean_shape"] = df_clean.shape

    # --- Étape 3 : Feature Engineering ---
    logger.info("━" * 60)
    logger.info("ÉTAPE 3/6 : Feature Engineering")
    logger.info("━" * 60)
    df_features = build_feature_set(df_clean)
    save_features(df_features)
    results["features_shape"] = df_features.shape
    results["new_features"] = [
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

    # --- Étape 4 : EDA ---
    logger.info("━" * 60)
    logger.info("ÉTAPE 4/6 : Analyse exploratoire (EDA)")
    logger.info("━" * 60)
    eda_results = run_eda(df_features)
    results["eda"] = eda_results

    # --- Étape 5 : Encodage + Validation ---
    logger.info("━" * 60)
    logger.info("ÉTAPE 5/6 : Encodage catégorielles + Validation qualité")
    logger.info("━" * 60)
    df_encoded = encode_categoricals(df_features)
    save_processed(df_encoded)

    quality_report = run_validation(df_encoded)
    save_quality_report(quality_report)
    results["quality"] = quality_report.to_dict()

    if not quality_report.passed:
        logger.error(
            "VALIDATION : %d test(s) critique(s) échoué(s). Vérifier quality_report.json",
            quality_report.n_failed,
        )
    else:
        logger.info("VALIDATION : tous les tests critiques sont réussis ✓")

    # --- Étape 6 : Dataset ML-ready ---
    logger.info("━" * 60)
    logger.info("ÉTAPE 6/6 : Préparation dataset ML-ready")
    logger.info("━" * 60)
    ml_result = build_ml_dataset(df_encoded)
    results["ml"] = {
        "n_features": len(ml_result["feature_names"]),
        "train_size": len(ml_result["X_train"]),
        "test_size": len(ml_result["X_test"]),
        "paths": ml_result["paths"],
    }

    # --- Résumé final ---
    logger.info("━" * 60)
    logger.info("PIPELINE JOUR 2 TERMINÉ")
    logger.info("━" * 60)
    logger.info("Lignes brutes        : %d", results["raw_shape"][0])
    logger.info("Lignes nettoyées     : %d", results["clean_shape"][0])
    logger.info("Features créées      : %d", len(results["new_features"]))
    logger.info("Colonnes ML-ready    : %d", results["ml"]["n_features"])
    logger.info("Train size           : %d", results["ml"]["train_size"])
    logger.info("Test size            : %d", results["ml"]["test_size"])
    logger.info("Taux décrochage      : %.1f%%", results["eda"]["stats"].get("dropout_rate_pct", 0))
    logger.info("Qualité données      : %d/%d tests réussis",
                results["quality"]["summary"]["passed"],
                results["quality"]["summary"]["total_tests"])
    logger.info("━" * 60)

    return results


if __name__ == "__main__":
    run_pipeline()
