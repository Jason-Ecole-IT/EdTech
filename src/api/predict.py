"""Module de prediction MLflow pour l'API FastAPI."""
import logging
import os
from typing import List

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = "edtech_dropout_classifier"
STAGE = "Production"

mlflow.set_tracking_uri(MLFLOW_URI)


class StudentFeatures(BaseModel):
    """Modele Pydantic pour les features d'un etudiant."""

    age: int = Field(..., description="Age de l'etudiant")
    travel_time: int = Field(..., description="Temps de trajet (1-4)")
    study_time: int = Field(..., description="Temps d'etude (1-4)")
    failures: int = Field(..., description="Nombre d'echecs")
    family_rel: int = Field(..., description="Relations familiales (1-5)")
    free_time: int = Field(..., description="Temps libre (1-5)")
    going_out: int = Field(..., description="Sorties (1-5)")
    weekday_alcohol: int = Field(..., description="Alcool semaine (1-5)")
    weekend_alcohol: int = Field(..., description="Alcool weekend (1-5)")
    health: int = Field(..., description="Sante (1-5)")
    absences: int = Field(..., description="Nombre d'absences")
    grade_1: int = Field(..., description="Note periode 1 (0-20)")
    grade_2: int = Field(..., description="Note periode 2 (0-20)")
    final_grade: int = Field(..., description="Note finale (0-20)")
    mother_edu: int = Field(..., description="Education mere (0-4)")
    father_edu: int = Field(..., description="Education pere (0-4)")
    mother_job: str = Field(..., description="Profession mere")
    father_job: str = Field(..., description="Profession pere")
    school: str = Field(..., description="Ecole (GP/MS)")
    sex: str = Field(..., description="Sexe (M/F)")
    address: str = Field(..., description="Adresse (U/R)")
    famsize: str = Field(..., description="Taille famille (LE3/GT3)")
    pstatus: str = Field(..., description="Statut parents (T/A)")
    schoolsup: str = Field(..., description="Soutien scolaire (yes/no)")
    famsup: str = Field(..., description="Soutien familial (yes/no)")
    paid: str = Field(..., description="Cours payants (yes/no)")
    activities: str = Field(..., description="Activites extra (yes/no)")
    nursery: str = Field(..., description="Ecole maternelle (yes/no)")
    higher: str = Field(..., description="Vise superieur (yes/no)")
    internet: str = Field(..., description="Internet (yes/no)")
    romantic: str = Field(..., description="Relation amoureuse (yes/no)")


class PredictionResponse(BaseModel):
    """Reponse de prediction."""

    student_id: str | None = None
    prediction: int = Field(..., description="0=Actif, 1=Decroche")
    probability: float = Field(..., description="Probabilite de decrochage")
    confidence: str = Field(..., description="Niveau de confiance")


def load_model_from_registry():
    """Charge le modele depuis MLflow Registry."""
    try:
        model_uri = f"models:/{MODEL_NAME}/{STAGE}"
        logger.info(f"Chargement du modele depuis {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Modele charge avec succes")
        return model
    except Exception as e:
        logger.error(f"Erreur chargement modele: {e}")
        raise


_model_cache = None


def get_model():
    """Retourne le modele (cache)."""
    global _model_cache
    if _model_cache is None:
        _model_cache = load_model_from_registry()
    return _model_cache


def build_feature_dict(features: StudentFeatures) -> dict:
    """Construit le dictionnaire de features pour le modele.

    Le modele attend 61 features. On construit les features engineered
    comme dans le pipeline Jour 2.
    """
    base = {
        "Age": features.age,
        "Travel_Time": features.travel_time,
        "Study_Time": features.study_time,
        "Number_of_Failures": features.failures,
        "Family_Relationship": features.family_rel,
        "Free_Time": features.free_time,
        "Going_Out": features.going_out,
        "Weekday_Alcohol_Consumption": features.weekday_alcohol,
        "Weekend_Alcohol_Consumption": features.weekend_alcohol,
        "Health_Status": features.health,
        "Number_of_Absences": features.absences,
        "Grade_1": features.grade_1,
        "Grade_2": features.grade_2,
        "Final_Grade": features.final_grade,
        "Mother_Education": features.mother_edu,
        "Father_Education": features.father_edu,
        "Mother_Job": features.mother_job,
        "Father_Job": features.father_job,
        "School": features.school,
        "Sex": features.sex,
        "Address": features.address,
        "Family_Size": features.famsize,
        "Parent_Status": features.pstatus,
        "School_Support": features.schoolsup,
        "Family_Support": features.famsup,
        "Paid_Classes": features.paid,
        "Extra_Curricular": features.activities,
        "Nursery": features.nursery,
        "Wants_Higher_Education": features.higher,
        "Internet_Access": features.internet,
        "Romantic_Relationship": features.romantic,
    }

    # Features engineered (simplifie pour l'API)
    avg_grade = (features.grade_1 + features.grade_2) / 2
    academic_score = (avg_grade + features.final_grade) / 2

    base.update(
        {
            "academic_performance_score": academic_score,
            "grade_progression": features.final_grade - avg_grade,
            "combined_risk_score": (
                features.failures * 2
                + features.absences * 0.5
                + (5 - features.family_rel) * 0.3
            ),
            "failure_history_flag": 1 if features.failures > 0 else 0,
            "engagement_score": features.study_time + (5 - features.free_time),
            "social_risk_index": features.weekday_alcohol + features.weekend_alcohol,
            "alcohol_risk_score": (features.weekday_alcohol + features.weekend_alcohol) / 2,
            "parental_education_level": (features.mother_edu + features.father_edu) / 2,
        }
    )

    return base


def predict_single(features: StudentFeatures) -> PredictionResponse:
    """Predit pour un etudiant."""
    model = get_model()
    feature_dict = build_feature_dict(features)
    df = pd.DataFrame([feature_dict])

    # Prediction
    pred_proba = model.predict(df)[0]
    pred = int(pred_proba > 0.5)

    # Confidence
    if pred_proba < 0.4:
        confidence = "Faible"
    elif pred_proba < 0.6:
        confidence = "Moyenne"
    elif pred_proba < 0.8:
        confidence = "Forte"
    else:
        confidence = "Tres forte"

    return PredictionResponse(
        prediction=pred,
        probability=float(pred_proba),
        confidence=confidence,
    )


def predict_batch(features_list: List[StudentFeatures]) -> List[PredictionResponse]:
    """Predit pour plusieurs etudiants."""
    model = get_model()
    feature_dicts = [build_feature_dict(f) for f in features_list]
    df = pd.DataFrame(feature_dicts)

    pred_probas = model.predict(df)
    predictions = []

    for i, prob in enumerate(pred_probas):
        pred = int(prob > 0.5)
        if prob < 0.4:
            conf = "Faible"
        elif prob < 0.6:
            conf = "Moyenne"
        elif prob < 0.8:
            conf = "Forte"
        else:
            conf = "Tres forte"

        predictions.append(
            PredictionResponse(
                student_id=str(i),
                prediction=pred,
                probability=float(prob),
                confidence=conf,
            )
        )

    return predictions
