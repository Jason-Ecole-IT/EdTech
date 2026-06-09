"""
Tests automatisés du pipeline Jour 2.

Couverture :
  - ETL (loader)
  - Feature engineering (features)
  - Validation qualité (validation)
  - Préparation ML (ml_dataset)
"""

import numpy as np
import pandas as pd
import pytest

from src.data.features import build_feature_set
from src.data.loader import clean, encode_categoricals
from src.data.ml_dataset import build_ml_dataset, select_features
from src.data.validation import run_validation


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def sample_raw() -> pd.DataFrame:
    """Dataset minimal reproduisant la structure réelle."""
    return pd.DataFrame({
        "School": ["GP", "MS", "GP"],
        "Gender": ["F", "M", "F"],
        "Age": [17, 18, 16],
        "Address": ["U", "R", "U"],
        "Family_Size": ["GT3", "LE3", "GT3"],
        "Parental_Status": ["T", "A", "T"],
        "Mother_Education": [4, 2, 3],
        "Father_Education": [4, 1, 2],
        "Mother_Job": ["teacher", "at_home", "other"],
        "Father_Job": ["other", "services", "teacher"],
        "Reason_for_Choosing_School": ["course", "home", "reputation"],
        "Guardian": ["mother", "father", "mother"],
        "Travel_Time": [2, 1, 3],
        "Study_Time": [3, 1, 2],
        "Number_of_Failures": [0, 1, 0],
        "School_Support": ["yes", "no", "yes"],
        "Family_Support": ["no", "yes", "no"],
        "Extra_Paid_Class": ["no", "no", "yes"],
        "Extra_Curricular_Activities": ["yes", "no", "no"],
        "Attended_Nursery": ["yes", "yes", "no"],
        "Wants_Higher_Education": ["yes", "no", "yes"],
        "Internet_Access": ["yes", "no", "yes"],
        "In_Relationship": ["no", "yes", "no"],
        "Family_Relationship": [4, 3, 5],
        "Free_Time": [3, 4, 2],
        "Going_Out": [2, 4, 3],
        "Weekend_Alcohol_Consumption": [1, 3, 2],
        "Weekday_Alcohol_Consumption": [1, 2, 1],
        "Health_Status": [4, 2, 5],
        "Number_of_Absences": [2, 12, 0],
        "Grade_1": [14, 8, 16],
        "Grade_2": [15, 7, 16],
        "Final_Grade": [15, 6, 17],
        "Dropped_Out": ["False", "True", "False"],
    })


@pytest.fixture
def sample_clean(sample_raw) -> pd.DataFrame:
    return clean(sample_raw)


@pytest.fixture
def sample_features(sample_clean) -> pd.DataFrame:
    return build_feature_set(sample_clean)


@pytest.fixture
def sample_encoded(sample_features) -> pd.DataFrame:
    return encode_categoricals(sample_features)


# ──────────────────────────────────────────────
# Tests ETL
# ──────────────────────────────────────────────

class TestClean:

    def test_dropped_out_is_integer(self, sample_clean):
        assert sample_clean["Dropped_Out"].dtype in [np.int64, np.int32, int]

    def test_binary_cols_are_0_or_1(self, sample_clean):
        binary_cols = ["School_Support", "Family_Support", "Internet_Access"]
        for col in binary_cols:
            assert set(sample_clean[col].unique()).issubset({0, 1}), f"{col} contient des valeurs hors {{0,1}}"

    def test_no_missing_values(self, sample_clean):
        assert sample_clean.isna().sum().sum() == 0

    def test_no_duplicates(self, sample_clean):
        assert sample_clean.duplicated().sum() == 0

    def test_grade_clipping(self):
        df = pd.DataFrame({
            "School": ["GP"], "Gender": ["F"], "Age": [17], "Address": ["U"],
            "Family_Size": ["GT3"], "Parental_Status": ["T"],
            "Mother_Education": [4], "Father_Education": [4],
            "Mother_Job": ["teacher"], "Father_Job": ["other"],
            "Reason_for_Choosing_School": ["course"], "Guardian": ["mother"],
            "Travel_Time": [1], "Study_Time": [2], "Number_of_Failures": [0],
            "School_Support": ["no"], "Family_Support": ["no"],
            "Extra_Paid_Class": ["no"], "Extra_Curricular_Activities": ["no"],
            "Attended_Nursery": ["yes"], "Wants_Higher_Education": ["yes"],
            "Internet_Access": ["yes"], "In_Relationship": ["no"],
            "Family_Relationship": [4], "Free_Time": [3], "Going_Out": [2],
            "Weekend_Alcohol_Consumption": [1], "Weekday_Alcohol_Consumption": [1],
            "Health_Status": [3], "Number_of_Absences": [50],
            "Grade_1": [25], "Grade_2": [-1], "Final_Grade": [15],
            "Dropped_Out": ["False"],
        })
        cleaned = clean(df)
        assert cleaned["Number_of_Absences"].max() <= 32
        assert cleaned["Grade_1"].max() <= 20
        assert cleaned["Grade_2"].min() >= 0


# ──────────────────────────────────────────────
# Tests Feature Engineering
# ──────────────────────────────────────────────

class TestFeatures:

    def test_all_features_created(self, sample_features):
        expected = [
            "academic_performance_score", "grade_progression", "alcohol_risk_score",
            "social_risk_index", "family_support_score", "engagement_score",
            "parental_education_level", "absence_risk_flag", "failure_history_flag",
            "combined_risk_score",
        ]
        for feat in expected:
            assert feat in sample_features.columns, f"Feature manquante : {feat}"

    def test_absence_risk_flag_logic(self, sample_features):
        high_absence = sample_features[sample_features["Number_of_Absences"] >= 10]
        assert (high_absence["absence_risk_flag"] == 1).all()

    def test_failure_history_flag_logic(self, sample_features):
        with_failure = sample_features[sample_features["Number_of_Failures"] > 0]
        assert (with_failure["failure_history_flag"] == 1).all()

    def test_social_risk_index_range(self, sample_features):
        assert sample_features["social_risk_index"].between(0, 1).all()

    def test_academic_performance_weighted(self, sample_features):
        row = sample_features.iloc[0]
        expected = round(row["Grade_1"] * 0.25 + row["Grade_2"] * 0.35 + row["Final_Grade"] * 0.40, 2)
        assert abs(row["academic_performance_score"] - expected) < 0.01

    def test_combined_risk_score_not_null(self, sample_features):
        assert sample_features["combined_risk_score"].isna().sum() == 0


# ──────────────────────────────────────────────
# Tests Validation
# ──────────────────────────────────────────────

class TestValidation:

    def test_validation_passes_on_clean_features(self, sample_encoded):
        report = run_validation(sample_encoded)
        critical_failures = [
            r for r in report.results
            if not r.passed and r.severity == "error"
            and r.test_name != "Dataset contient au moins 100 lignes"
        ]
        assert len(critical_failures) == 0, \
            f"Tests critiques échoués : {[r.test_name for r in critical_failures]}"

    def test_report_has_summary(self, sample_encoded):
        report = run_validation(sample_encoded)
        d = report.to_dict()
        assert "summary" in d
        assert "total_tests" in d["summary"]
        assert d["summary"]["total_tests"] > 0


# ──────────────────────────────────────────────
# Tests ML Dataset
# ──────────────────────────────────────────────

class TestMLDataset:

    @pytest.fixture
    def large_encoded(self) -> pd.DataFrame:
        """Dataset suffisamment large pour le split stratifié."""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "Age": np.random.randint(15, 22, n),
            "Study_Time": np.random.randint(1, 4, n),
            "Number_of_Failures": np.random.randint(0, 3, n),
            "Number_of_Absences": np.random.randint(0, 20, n),
            "Grade_1": np.random.randint(0, 20, n),
            "Grade_2": np.random.randint(0, 20, n),
            "Final_Grade": np.random.randint(0, 20, n),
            "Mother_Education": np.random.randint(0, 4, n),
            "Father_Education": np.random.randint(0, 4, n),
            "Family_Relationship": np.random.randint(1, 5, n),
            "Going_Out": np.random.randint(1, 5, n),
            "Weekend_Alcohol_Consumption": np.random.randint(1, 5, n),
            "Weekday_Alcohol_Consumption": np.random.randint(1, 5, n),
            "Health_Status": np.random.randint(1, 5, n),
            "Free_Time": np.random.randint(1, 5, n),
            "Travel_Time": np.random.randint(1, 4, n),
            "School_Support": np.random.randint(0, 1, n),
            "Family_Support": np.random.randint(0, 1, n),
            "Wants_Higher_Education": np.random.randint(0, 1, n),
            "Internet_Access": np.random.randint(0, 1, n),
            "academic_performance_score": np.random.uniform(0, 20, n),
            "grade_progression": np.random.uniform(-5, 5, n),
            "alcohol_risk_score": np.random.uniform(1, 5, n),
            "social_risk_index": np.random.uniform(0, 1, n),
            "family_support_score": np.random.uniform(0, 10, n),
            "engagement_score": np.random.uniform(0, 5, n),
            "parental_education_level": np.random.uniform(0, 4, n),
            "absence_risk_flag": np.random.randint(0, 1, n),
            "failure_history_flag": np.random.randint(0, 1, n),
            "combined_risk_score": np.random.uniform(0, 1, n),
            "Dropped_Out": np.random.randint(0, 1, n),
        })
        return df

    def test_split_sizes(self, large_encoded):
        result = build_ml_dataset(large_encoded, test_size=0.2)
        total = len(result["X_train"]) + len(result["X_test"])
        assert total == len(large_encoded)

    def test_no_target_in_features(self, large_encoded):
        result = build_ml_dataset(large_encoded)
        assert "Dropped_Out" not in result["X_train"].columns

    def test_feature_names_consistent(self, large_encoded):
        result = build_ml_dataset(large_encoded)
        assert list(result["X_train"].columns) == result["feature_names"]
        assert list(result["X_test"].columns) == result["feature_names"]
