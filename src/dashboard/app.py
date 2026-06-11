import os
import requests
import streamlit as st
import pandas as pd

st.set_page_config(page_title="EdTech Dropout Prediction", layout="wide", page_icon="🎓")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🎓 EdTech Dropout Prediction</h1>
    <p style="color: white; opacity: 0.9; margin: 0.5rem 0 0 0;">Predict student dropout risk with AI</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Single Student", "Batch Upload"])

with tab1:
    st.markdown("### Predict for a Single Student")

    with st.form("single_prediction"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 15, 30, 18)
            school = st.selectbox("School", ["GP", "MS"])
            sex = st.selectbox("Sex", ["M", "F"])
            grade_1 = st.number_input("Grade Period 1", 0, 20, 14)
            grade_2 = st.number_input("Grade Period 2", 0, 20, 15)
            final_grade = st.number_input("Final Grade", 0, 20, 16)
        with col2:
            failures = st.number_input("Past Failures", 0, 10, 0)
            absences = st.number_input("Absences", 0, 93, 2)
            study_time = st.slider("Study Time (1-4)", 1, 4, 3)
            travel_time = st.slider("Travel Time (1-4)", 1, 4, 2)
            weekday_alcohol = st.slider("Weekday Alcohol (1-5)", 1, 5, 1)
            weekend_alcohol = st.slider("Weekend Alcohol (1-5)", 1, 5, 2)

        submitted = st.form_submit_button("🔍 Predict Dropout Risk", type="primary")

        if submitted:
            payload = {
                "age": age, "travel_time": travel_time, "study_time": study_time,
                "failures": failures, "family_rel": 4, "free_time": 3, "going_out": 3,
                "weekday_alcohol": weekday_alcohol, "weekend_alcohol": weekend_alcohol,
                "health": 4, "absences": absences, "grade_1": grade_1,
                "grade_2": grade_2, "final_grade": final_grade, "mother_edu": 2,
                "father_edu": 2, "mother_job": "services", "father_job": "other",
                "school": school, "sex": sex, "address": "U", "famsize": "GT3",
                "pstatus": "T", "schoolsup": "no", "famsup": "yes", "paid": "no",
                "activities": "yes", "nursery": "yes", "higher": "yes",
                "internet": "yes", "romantic": "no",
            }
            try:
                r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                r.raise_for_status()
                result = r.json()
                if result["prediction"] == 1:
                    st.markdown(f"""
                    <div class="prediction-result">
                        <h2 style="color: #ff6b6b;">🔴 High Dropout Risk</h2>
                        <p style="font-size: 1.5rem;">Probability: {result['probability']*100:.1f}%</p>
                        <p>Confidence: {result['confidence']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-result">
                        <h2 style="color: #51cf66;">🟢 Low Dropout Risk</h2>
                        <p style="font-size: 1.5rem;">Probability: {result['probability']*100:.1f}%</p>
                        <p>Confidence: {result['confidence']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")

with tab2:
    st.markdown("### Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} students")
            if st.button("🚀 Predict All", type="primary"):
                with st.spinner("Predicting..."):
                    students = df.to_dict("records")
                    for student in students:
                        student.setdefault("family_rel", 4)
                        student.setdefault("free_time", 3)
                        student.setdefault("going_out", 3)
                        student.setdefault("health", 4)
                        student.setdefault("mother_edu", 2)
                        student.setdefault("father_edu", 2)
                        student.setdefault("mother_job", "services")
                        student.setdefault("father_job", "other")
                        student.setdefault("pstatus", "T")
                        student.setdefault("schoolsup", "no")
                        student.setdefault("famsup", "yes")
                        student.setdefault("paid", "no")
                        student.setdefault("activities", "yes")
                        student.setdefault("nursery", "yes")
                        student.setdefault("internet", "yes")
                        student.setdefault("romantic", "no")
                    try:
                        r = requests.post(f"{API_URL}/batch_predict", json=students, timeout=30)
                        r.raise_for_status()
                        results = r.json()
                        df["prediction"] = ["Dropout" if r["prediction"] == 1 else "Active" for r in results]
                        df["probability"] = [f"{r['probability']*100:.1f}%" for r in results]
                        df["confidence"] = [r["confidence"] for r in results]
                        st.success("Predictions completed!")
                        st.dataframe(df, use_container_width=True)
                        csv = df.to_csv(index=False)
                        st.download_button("� Download Results", csv, "predictions.csv", "text/csv")
                    except Exception as e:
                        st.error(f"Batch prediction error: {e}")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        st.info("Upload a CSV file with student data.")
        st.markdown("**Required columns:** age, school, sex, grade_1, grade_2, final_grade, failures, absences, study_time, travel_time, weekday_alcohol, weekend_alcohol")
