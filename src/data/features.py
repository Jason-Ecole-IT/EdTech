"""
Feature Engineering - Création de variables métier pertinentes.

Toutes les features sont justifiées dans le contexte du décrochage scolaire.
Ce module s'applique sur le DataFrame nettoyé AVANT encodage one-hot.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")


def add_academic_performance_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature : academic_performance_score
    Justification : Moyenne pondérée des trois notes (G1, G2 comptent moins
    que la note finale). Un score faible est le signal le plus direct de décrochage.
    """
    df["academic_performance_score"] = (
        df["Grade_1"] * 0.25 + df["Grade_2"] * 0.35 + df["Final_Grade"] * 0.40
    ).round(2)
    return df


def add_grade_progression(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature : grade_progression
    Justification : Évolution entre G1 et G2. Une progression négative
    indique une détérioration des résultats en cours d'année, signe précurseur
    de décrochage. Progression positive = dynamique favorable.
    """
    df["grade_progression"] = (df["Grade_2"] - df["Grade_1"]).round(2)
    return df


def add_alcohol_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature : alcohol_risk_score
    Justification : Somme pondérée consommation weekend (moins critique)
    et semaine (plus critique car impact sur les cours). Score > 5 = signal
    de comportement à risque fort.
    """
    df["alcohol_risk_score"] = (
        df["Weekday_Alcohol_Consumption"] * 0.6
        + df["Weekend_Alcohol_Consumption"] * 0.4
    ).round(2)
    return df


def add_social_risk_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature : social_risk_index
    Justification : Indice combinant sorties fréquentes, alcool et temps libre
    non structuré. Un indice élevé corrèle avec un désengagement scolaire.
    Normalisé entre 0 et 1.
    """
    raw = (
        df["Going_Out"] * 0.4
        + df["alcohol_risk_score"] * 0.4
        + df["Free_Time"] * 0.2
    )
    df["social_risk_index"] = ((raw - raw.min()) / (raw.max() - raw.min())).round(3)
    return df


def add_family_support_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature : family_support_score
    Justification : Score composite du soutien familial. Inclut la qualité
    des relations familiales (1-5), le soutien de l'école et de la famille.
    Un score élevé est un facteur protecteur contre le décrochage.
    """
    school_support = df.get("School_Support", pd.Series(0, index=df.index))
    family_support = df.get("Family_Support", pd.Series(0, index=df.index))

    df["family_support_score"] = (
        df["Family_Relationship"] * 0.5
        + family_support * 2.0
        + school_support * 1.5
    ).round(2)
    return df


def add_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature : engagement_score
    Justification : Score d'engagement scolaire basé sur le temps d'étude,
    les activités extra-scolaires, l'aspiration à l'enseignement supérieur
    et l'accès à internet. Plus ce score est élevé, plus l'étudiant est
    impliqué dans sa scolarité.
    """
    wants_higher = df.get("Wants_Higher_Education", pd.Series(0, index=df.index))
    internet = df.get("Internet_Access", pd.Series(0, index=df.index))
    extra = df.get("Extra_Curricular_Activities", pd.Series(0, index=df.index))

    df["engagement_score"] = (
        df["Study_Time"] * 0.4
        + wants_higher * 1.5
        + internet * 0.5
        + extra * 0.5
    ).round(2)
    return df


def add_parental_education_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature : parental_education_level
    Justification : Niveau d'éducation des parents (0=aucun à 4=supérieur).
    La moyenne des deux parents est un proxy du capital culturel du foyer,
    fortement corrélé à la réussite scolaire.
    """
    df["parental_education_level"] = (
        (df["Mother_Education"] + df["Father_Education"]) / 2
    ).round(2)
    return df


def add_absence_risk_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature : absence_risk_flag
    Justification : Flag binaire activé si l'étudiant dépasse le seuil critique
    de 10 absences. Au-delà de ce seuil, la probabilité de décrochage augmente
    fortement d'après la littérature sur le sujet.
    """
    df["absence_risk_flag"] = (df["Number_of_Absences"] >= 10).astype(int)
    return df


def add_failure_history_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature : failure_history_flag
    Justification : Flag binaire indiquant si l'étudiant a déjà au moins un
    échec antérieur. Un historique d'échec est le prédicteur le plus fort
    du décrochage dans les modèles de classification.
    """
    df["failure_history_flag"] = (df["Number_of_Failures"] > 0).astype(int)
    return df


def add_combined_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature : combined_risk_score
    Justification : Score de risque global agrégé à partir de tous les
    indicateurs de risque. Utile comme feature synthétique et comme
    métrique de monitoring pédagogique.
    """
    perf_norm = 1 - (df["academic_performance_score"] / 20).clip(0, 1)

    df["combined_risk_score"] = (
        perf_norm * 0.35
        + df["social_risk_index"] * 0.20
        + df["absence_risk_flag"] * 0.20
        + df["failure_history_flag"] * 0.15
        + (1 - (df["engagement_score"] / df["engagement_score"].max())) * 0.10
    ).round(3)
    return df


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """Applique séquentiellement toutes les transformations de features."""
    logger.info("=== Feature engineering : début ===")
    df = add_academic_performance_score(df)
    df = add_grade_progression(df)
    df = add_alcohol_risk_score(df)
    df = add_social_risk_index(df)
    df = add_family_support_score(df)
    df = add_engagement_score(df)
    df = add_parental_education_level(df)
    df = add_absence_risk_flag(df)
    df = add_failure_history_flag(df)
    df = add_combined_risk_score(df)

    new_features = [
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
    logger.info("Features créées : %s", new_features)
    logger.info("=== Feature engineering : terminé (%d colonnes) ===", len(df.columns))
    return df


def save_features(df: pd.DataFrame, filename: str = "students_features.parquet") -> Path:
    """Sauvegarde le dataset avec features dans data/features/."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FEATURES_DIR / filename
    df.to_parquet(out, index=False)
    logger.info("Dataset features sauvegardé : %s (%d lignes, %d colonnes)", out, len(df), len(df.columns))
    return out
