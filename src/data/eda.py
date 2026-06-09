"""
EDA - Analyse exploratoire des données et génération de rapport.

Produit :
  - data/processed/eda_report.json  : statistiques et insights
  - data/processed/eda_plots/       : graphiques PNG
"""

import json
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
PLOTS_DIR = PROCESSED_DIR / "eda_plots"


def _save_fig(fig, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / f"{name}.png"
    try:
        fig.write_image(str(path))
        logger.info("Graphique sauvegardé : %s", path)
    except Exception as exc:
        logger.warning("Impossible de sauvegarder %s : %s", name, exc)


def descriptive_stats(df: pd.DataFrame) -> dict:
    """Calcule les statistiques descriptives clés."""
    target_col = "Dropped_Out"
    dropout_rate = round(df[target_col].mean() * 100, 2) if target_col in df.columns else None

    num_df = df.select_dtypes(include="number")
    stats = {
        "n_students": len(df),
        "n_features": len(df.columns),
        "dropout_rate_pct": dropout_rate,
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "numeric_summary": num_df.describe().round(2).to_dict(),
    }
    logger.info(
        "Statistiques : %d étudiants, taux décrochage=%.1f%%",
        stats["n_students"],
        stats["dropout_rate_pct"] or 0,
    )
    return stats


def plot_dropout_distribution(df: pd.DataFrame) -> None:
    """Distribution de la variable cible."""
    if "Dropped_Out" not in df.columns:
        return
    counts = df["Dropped_Out"].value_counts().reset_index()
    counts.columns = ["Dropped_Out", "count"]
    counts["label"] = counts["Dropped_Out"].map({0: "Actif", 1: "Décroché"})
    fig = px.bar(
        counts,
        x="label",
        y="count",
        color="label",
        color_discrete_map={"Actif": "#2ecc71", "Décroché": "#e74c3c"},
        title="Distribution des étudiants : Actifs vs Décrochés",
        labels={"count": "Nombre d'étudiants", "label": "Statut"},
        text="count",
    )
    fig.update_layout(showlegend=False)
    _save_fig(fig, "01_dropout_distribution")


def plot_grade_distributions(df: pd.DataFrame) -> None:
    """Distribution des trois notes selon le statut de décrochage."""
    grade_cols = [c for c in ["Grade_1", "Grade_2", "Final_Grade"] if c in df.columns]
    if not grade_cols or "Dropped_Out" not in df.columns:
        return

    fig = make_subplots(rows=1, cols=len(grade_cols), subplot_titles=grade_cols)
    colors = {0: "#2ecc71", 1: "#e74c3c"}
    labels = {0: "Actif", 1: "Décroché"}

    for i, col in enumerate(grade_cols, start=1):
        for status in [0, 1]:
            subset = df[df["Dropped_Out"] == status][col]
            fig.add_trace(
                go.Histogram(
                    x=subset,
                    name=labels[status],
                    marker_color=colors[status],
                    opacity=0.7,
                    showlegend=(i == 1),
                ),
                row=1,
                col=i,
            )
    fig.update_layout(
        title="Distribution des notes par statut de décrochage",
        barmode="overlay",
        height=400,
    )
    _save_fig(fig, "02_grade_distributions")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Heatmap de corrélation des variables numériques."""
    num_cols = [
        "Grade_1", "Grade_2", "Final_Grade",
        "Number_of_Absences", "Number_of_Failures",
        "Study_Time", "Going_Out", "Weekend_Alcohol_Consumption",
        "Weekday_Alcohol_Consumption", "Health_Status",
        "academic_performance_score", "combined_risk_score",
        "engagement_score", "social_risk_index", "Dropped_Out",
    ]
    available = [c for c in num_cols if c in df.columns]
    corr = df[available].corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Matrice de corrélation",
        aspect="auto",
    )
    fig.update_layout(height=600)
    _save_fig(fig, "03_correlation_heatmap")


def plot_absences_vs_dropout(df: pd.DataFrame) -> None:
    """Boxplot absences vs décrochage."""
    if "Number_of_Absences" not in df.columns or "Dropped_Out" not in df.columns:
        return
    df_plot = df.copy()
    df_plot["Statut"] = df_plot["Dropped_Out"].map({0: "Actif", 1: "Décroché"})
    fig = px.box(
        df_plot,
        x="Statut",
        y="Number_of_Absences",
        color="Statut",
        color_discrete_map={"Actif": "#2ecc71", "Décroché": "#e74c3c"},
        title="Absences selon le statut de décrochage",
        points="outliers",
    )
    _save_fig(fig, "04_absences_vs_dropout")


def plot_study_time_vs_dropout(df: pd.DataFrame) -> None:
    """Temps d'étude et décrochage."""
    if "Study_Time" not in df.columns or "Dropped_Out" not in df.columns:
        return
    df_plot = df.copy()
    df_plot["Statut"] = df_plot["Dropped_Out"].map({0: "Actif", 1: "Décroché"})
    study_labels = {1: "<2h", 2: "2-5h", 3: "5-10h", 4: ">10h"}
    df_plot["Study_Label"] = df_plot["Study_Time"].map(study_labels)

    grouped = df_plot.groupby(["Study_Label", "Statut"]).size().reset_index(name="count")
    fig = px.bar(
        grouped,
        x="Study_Label",
        y="count",
        color="Statut",
        barmode="group",
        color_discrete_map={"Actif": "#2ecc71", "Décroché": "#e74c3c"},
        title="Temps d'étude hebdomadaire vs décrochage",
        category_orders={"Study_Label": ["<2h", "2-5h", "5-10h", ">10h"]},
    )
    _save_fig(fig, "05_study_time_vs_dropout")


def plot_risk_score_distribution(df: pd.DataFrame) -> None:
    """Distribution du score de risque combiné."""
    if "combined_risk_score" not in df.columns or "Dropped_Out" not in df.columns:
        return
    df_plot = df.copy()
    df_plot["Statut"] = df_plot["Dropped_Out"].map({0: "Actif", 1: "Décroché"})
    fig = px.histogram(
        df_plot,
        x="combined_risk_score",
        color="Statut",
        nbins=30,
        barmode="overlay",
        opacity=0.75,
        color_discrete_map={"Actif": "#2ecc71", "Décroché": "#e74c3c"},
        title="Distribution du score de risque combiné",
    )
    _save_fig(fig, "06_risk_score_distribution")


def compute_insights(df: pd.DataFrame) -> dict:
    """Génère les insights clés du dataset pour le rapport."""
    insights = {}

    if "Dropped_Out" in df.columns:
        insights["dropout_rate_pct"] = round(df["Dropped_Out"].mean() * 100, 2)

    if "Number_of_Failures" in df.columns and "Dropped_Out" in df.columns:
        dropout_with_failure = df[df["Number_of_Failures"] > 0]["Dropped_Out"].mean()
        insights["dropout_rate_with_failure_pct"] = round(dropout_with_failure * 100, 2)

    if "absence_risk_flag" in df.columns and "Dropped_Out" in df.columns:
        dropout_high_absence = df[df["absence_risk_flag"] == 1]["Dropped_Out"].mean()
        insights["dropout_rate_high_absence_pct"] = round(dropout_high_absence * 100, 2)

    if "Final_Grade" in df.columns and "Dropped_Out" in df.columns:
        avg_grade_dropout = df[df["Dropped_Out"] == 1]["Final_Grade"].mean()
        avg_grade_active = df[df["Dropped_Out"] == 0]["Final_Grade"].mean()
        insights["avg_final_grade_dropout"] = round(avg_grade_dropout, 2)
        insights["avg_final_grade_active"] = round(avg_grade_active, 2)

    if "combined_risk_score" in df.columns and "Dropped_Out" in df.columns:
        corr = df[["combined_risk_score", "Dropped_Out"]].corr().iloc[0, 1]
        insights["combined_risk_score_correlation"] = round(corr, 3)

    logger.info("Insights calculés : %s", insights)
    return insights


def save_eda_report(stats: dict, insights: dict) -> Path:
    """Sauvegarde le rapport EDA en JSON."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    report = {"statistics": stats, "insights": insights}
    out = PROCESSED_DIR / "eda_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Rapport EDA sauvegardé : %s", out)
    return out


def run_eda(df: pd.DataFrame) -> dict:
    """Exécute l'analyse EDA complète et génère les graphiques + rapport."""
    logger.info("=== EDA Jour 2 : début ===")
    stats = descriptive_stats(df)
    insights = compute_insights(df)

    plot_dropout_distribution(df)
    plot_grade_distributions(df)
    plot_correlation_heatmap(df)
    plot_absences_vs_dropout(df)
    plot_study_time_vs_dropout(df)
    plot_risk_score_distribution(df)

    save_eda_report(stats, insights)
    logger.info("=== EDA Jour 2 : terminé ===")
    return {"stats": stats, "insights": insights}
