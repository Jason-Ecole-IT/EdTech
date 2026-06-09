"""
ETL - Chargement, nettoyage et sauvegarde des données LMS.

Dataset : Student Dropout (649 étudiants, 34 colonnes)
Cible   : Dropped_Out (bool) -> risque de décrochage
"""

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

BINARY_COLS = [
    "School_Support",
    "Family_Support",
    "Extra_Paid_Class",
    "Extra_Curricular_Activities",
    "Attended_Nursery",
    "Wants_Higher_Education",
    "Internet_Access",
    "In_Relationship",
]

ORDINAL_EDUCATION = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

EXPECTED_RANGES = {
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
}


def load_raw(filename: str = "student dropout.csv") -> pd.DataFrame:
    """Charge le fichier CSV brut depuis data/raw/."""
    filepath = RAW_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")

    df = pd.read_csv(filepath)
    logger.info("Données brutes chargées : %d lignes, %d colonnes", len(df), len(df.columns))
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage complet du dataset."""
    df = df.copy()

    # --- Noms de colonnes normalisés ---
    df.columns = [c.strip() for c in df.columns]

    # --- Suppression des doublons ---
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        logger.warning("%d doublon(s) supprimé(s)", removed)
    else:
        logger.info("Aucun doublon détecté")

    # --- Cible : booléen -> entier ---
    df["Dropped_Out"] = df["Dropped_Out"].map(
        lambda x: 1 if str(x).strip().lower() in ("true", "1", "yes") else 0
    )

    # --- Colonnes binaires yes/no -> 0/1 ---
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({"yes": 1, "no": 0})
            if df[col].isna().any():
                logger.warning("Valeurs inattendues dans '%s', remplacement par 0", col)
                df[col] = df[col].fillna(0).astype(int)

    # --- Correction des types numériques ---
    numeric_cols = list(EXPECTED_RANGES.keys()) + [
        "Mother_Education",
        "Father_Education",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Valeurs manquantes : médiane pour numériques, mode pour catégorielles ---
    num_missing = df[numeric_cols].isna().sum()
    for col, cnt in num_missing.items():
        if cnt > 0:
            median = df[col].median()
            df[col] = df[col].fillna(median)
            logger.warning("'%s' : %d valeur(s) manquante(s) remplacée(s) par médiane=%.1f", col, cnt, median)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        cnt = df[col].isna().sum()
        if cnt > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.warning("'%s' : %d valeur(s) manquante(s) remplacée(s) par mode='%s'", col, cnt, mode_val)

    # --- Correction des incohérences de grades ---
    # Grade_1=0 ET Grade_2=0 ET Final_Grade>0 : Grade_1/2 probablement manquants
    mask_zero_grades = (df["Grade_1"] == 0) & (df["Grade_2"] == 0) & (df["Final_Grade"] > 0)
    if mask_zero_grades.sum() > 0:
        logger.warning(
            "%d étudiant(s) avec Grade_1=0 et Grade_2=0 mais Final_Grade>0 détectés",
            mask_zero_grades.sum(),
        )

    # --- Clipping des valeurs hors plage attendue ---
    for col, (lo, hi) in EXPECTED_RANGES.items():
        if col in df.columns:
            out = ((df[col] < lo) | (df[col] > hi)).sum()
            if out:
                logger.warning("'%s' : %d valeur(s) hors plage [%d, %d] -> clipping", col, out, lo, hi)
            df[col] = df[col].clip(lo, hi)

    logger.info("Nettoyage terminé : %d lignes, %d colonnes", len(df), len(df.columns))
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encodage des variables catégorielles restantes (one-hot)."""
    df = df.copy()

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if not cat_cols:
        logger.info("Aucune variable catégorielle à encoder")
        return df

    logger.info("Encodage one-hot : %s", cat_cols)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)
    return df


def save_processed(df: pd.DataFrame, filename: str = "students_clean.parquet") -> Path:
    """Sauvegarde les données nettoyées dans data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / filename
    df.to_parquet(out, index=False)
    logger.info("Données nettoyées sauvegardées : %s (%d lignes)", out, len(df))
    return out


def run_etl() -> pd.DataFrame:
    """Exécute la chaîne ETL complète : load -> clean -> encode -> save."""
    logger.info("=== ETL Jour 2 : début ===")
    df_raw = load_raw()
    df_clean = clean(df_raw)
    df_encoded = encode_categoricals(df_clean)
    save_processed(df_encoded)
    logger.info("=== ETL Jour 2 : terminé ===")
    return df_encoded
