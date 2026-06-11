"""Tests pour l'API FastAPI Jour 4."""
import os

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.predict import StudentFeatures, predict_single

client = TestClient(app)


@pytest.fixture
def sample_student():
    """Donnees etudiant de test."""
    return StudentFeatures(
        age=18,
        travel_time=2,
        study_time=3,
        failures=0,
        family_rel=4,
        free_time=3,
        going_out=3,
        weekday_alcohol=1,
        weekend_alcohol=2,
        health=4,
        absences=2,
        grade_1=14,
        grade_2=15,
        final_grade=16,
        mother_edu=2,
        father_edu=2,
        mother_job="services",
        father_job="other",
        school="GP",
        sex="M",
        address="U",
        famsize="GT3",
        pstatus="T",
        schoolsup="no",
        famsup="yes",
        paid="no",
        activities="yes",
        nursery="yes",
        higher="yes",
        internet="yes",
        romantic="no",
    )


class TestHealthEndpoint:
    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["project"] == "EdTech Analytics Pedagogique"
        assert data["status"] == "running"

    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert "api" in data
        assert "database" in data
        assert "mlflow" in data
        assert "timestamp" in data


class TestPredictEndpoint:
    def test_predict_success(self, sample_student):
        r = client.post("/predict", json=sample_student.model_dump())
        assert r.status_code == 200
        data = r.json()
        assert "prediction" in data
        assert "probability" in data
        assert "confidence" in data
        assert data["prediction"] in [0, 1]
        assert 0 <= data["probability"] <= 1

    def test_predict_invalid_data(self):
        invalid = {"age": "invalid"}
        r = client.post("/predict", json=invalid)
        assert r.status_code in [422, 400]

    def test_predict_missing_field(self):
        incomplete = {"age": 18, "travel_time": 2}
        r = client.post("/predict", json=incomplete)
        assert r.status_code in [422, 400]


class TestBatchPredictEndpoint:
    def test_batch_predict_success(self, sample_student):
        students = [sample_student.model_dump(), sample_student.model_dump()]
        r = client.post("/batch_predict", json=students)
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2
        for pred in data:
            assert "prediction" in pred
            assert "probability" in pred
            assert "confidence" in pred

    def test_batch_predict_empty(self):
        r = client.post("/batch_predict", json=[])
        assert r.status_code == 422 or r.status_code == 400

    def test_batch_predict_single(self, sample_student):
        r = client.post("/batch_predict", json=[sample_student.model_dump()])
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1


class TestPredictModule:
    def test_predict_single(self, sample_student):
        result = predict_single(sample_student)
        assert result.prediction in [0, 1]
        assert 0 <= result.probability <= 1
        assert result.confidence in ["Faible", "Moyenne", "Forte", "Tres forte"]

    def test_feature_dict_building(self, sample_student):
        from src.api.predict import build_feature_dict

        feature_dict = build_feature_dict(sample_student)
        assert isinstance(feature_dict, dict)
        assert "academic_performance_score" in feature_dict
        assert "combined_risk_score" in feature_dict
        assert "engagement_score" in feature_dict
