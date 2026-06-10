"""
Comparaison et sélection du meilleur modèle.

Critères : F1-score (prioritaire pour dataset déséquilibré),
           ROC-AUC, accuracy, temps d'inférence.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")

METRIC_WEIGHTS = {
    "f1": 0.40,
    "roc_auc": 0.30,
    "accuracy": 0.20,
    "recall": 0.10,
}


def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    """Construit un DataFrame de comparaison des modèles."""
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append({
            "model": r["name"],
            "accuracy": m.get("accuracy"),
            "f1": m.get("f1"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "roc_auc": m.get("roc_auc"),
            "train_time_s": m.get("train_time_s"),
            "inference_time_ms": m.get("inference_time_ms"),
            "run_id": r.get("run_id"),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("f1", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


def compute_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score composite pondéré pour sélection objective.
    Normalise chaque métrique entre 0 et 1 avant pondération.
    """
    df = df.copy()
    for metric, weight in METRIC_WEIGHTS.items():
        if metric in df.columns and df[metric].notna().any():
            col_min = df[metric].min()
            col_max = df[metric].max()
            if col_max > col_min:
                df[f"{metric}_norm"] = (df[metric] - col_min) / (col_max - col_min)
            else:
                df[f"{metric}_norm"] = 1.0

    df["composite_score"] = sum(
        df.get(f"{m}_norm", 0) * w
        for m, w in METRIC_WEIGHTS.items()
        if f"{m}_norm" in df.columns
    )
    df["composite_score"] = df["composite_score"].round(4)
    return df


def select_best_model(results: list[dict], primary_metric: str = "f1") -> dict:
    """
    Sélectionne le meilleur modèle.

    Justification du choix de F1 comme métrique primaire :
    - Dataset déséquilibré (85% actifs, 15% décrochés)
    - L'accuracy seule serait trompeuse (prédire toujours 'actif' = 85% accuracy)
    - F1 équilibre précision et rappel, adapté à la détection de décrochage
    - ROC-AUC complète en mesurant la discrimination globale
    """
    df = build_comparison_table(results)
    df = compute_composite_score(df)

    best_idx = df["composite_score"].idxmax()
    best_row = df.loc[best_idx]
    best_name = best_row["model"]

    best_result = next(r for r in results if r["name"] == best_name)

    logger.info("━" * 50)
    logger.info("COMPARAISON DES MODÈLES")
    logger.info("━" * 50)
    for _, row in df.iterrows():
        marker = " ← BEST" if row["model"] == best_name else ""
        logger.info(
            "%-30s | F1=%.4f | ROC=%.4f | Score=%.4f%s",
            row["model"],
            row.get("f1") or 0,
            row.get("roc_auc") or 0,
            row.get("composite_score") or 0,
            marker,
        )
    logger.info("━" * 50)
    logger.info("MEILLEUR MODÈLE : %s (composite_score=%.4f)", best_name, best_row["composite_score"])

    return {
        "best_result": best_result,
        "comparison_df": df,
        "selection_metric": primary_metric,
        "justification": (
            f"'{best_name}' sélectionné avec F1={best_row['f1']:.4f}, "
            f"ROC-AUC={best_row.get('roc_auc') or 0:.4f}, "
            f"score composite={best_row['composite_score']:.4f}. "
            "F1 priorisé car dataset déséquilibré (15% décrochage)."
        ),
    }


def save_comparison_report(df: pd.DataFrame, justification: str) -> Path:
    """Sauvegarde le rapport de comparaison."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = PROCESSED_DIR / "model_comparison.csv"
    out_txt = PROCESSED_DIR / "model_selection_justification.txt"

    df.to_csv(out_csv, index=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(justification + "\n\n")
        f.write(df.to_string())

    logger.info("Rapport comparaison sauvegardé : %s", out_csv)
    return out_csv
