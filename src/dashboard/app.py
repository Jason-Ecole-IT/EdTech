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

            column_mapping = {
                "School": "school",
                "Gender": "sex",
                "Age": "age",
                "Grade_1": "grade_1",
                "Grade_2": "grade_2",
                "Final_Grade": "final_grade",
                "Number_of_Failures": "failures",
                "Number_of_Absences": "absences",
                "Study_Time": "study_time",
                "Travel_Time": "travel_time",
                "Weekday_Alcohol_Consumption": "weekday_alcohol",
                "Weekend_Alcohol_Consumption": "weekend_alcohol",
            }

            df = df.rename(columns=column_mapping)

            required_cols = ["age", "school", "sex", "grade_1", "grade_2", "final_grade", "failures", "absences", "study_time", "travel_time", "weekday_alcohol", "weekend_alcohol"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                st.success("All required columns found!")
                sample_size = st.number_input("Sample size (for testing)", min_value=1, max_value=len(df), value=10)
                if st.button("🚀 Predict All", type="primary"):
                    with st.spinner("Predicting..."):
                        students = df.head(sample_size).to_dict("records")
                        cleaned_students = []
                        for student in students:
                            cleaned = {
                                "age": student.get("age", 18),
                                "travel_time": student.get("travel_time", 2),
                                "study_time": student.get("study_time", 3),
                                "failures": student.get("failures", 0),
                                "family_rel": student.get("family_rel", student.get("Family_Relationship", 4)),
                                "free_time": student.get("free_time", student.get("Free_Time", 3)),
                                "going_out": student.get("going_out", student.get("Going_Out", 3)),
                                "weekday_alcohol": student.get("weekday_alcohol", student.get("Weekday_Alcohol_Consumption", 1)),
                                "weekend_alcohol": student.get("weekend_alcohol", student.get("Weekend_Alcohol_Consumption", 1)),
                                "health": student.get("health", student.get("Health_Status", 4)),
                                "absences": student.get("absences", student.get("Number_of_Absences", 0)),
                                "grade_1": student.get("grade_1", student.get("Grade_1", 14)),
                                "grade_2": student.get("grade_2", student.get("Grade_2", 15)),
                                "final_grade": student.get("final_grade", student.get("Final_Grade", 16)),
                                "mother_edu": student.get("mother_edu", student.get("Mother_Education", 2)),
                                "father_edu": student.get("father_edu", student.get("Father_Education", 2)),
                                "mother_job": student.get("mother_job", student.get("Mother_Job", "services")),
                                "father_job": student.get("father_job", student.get("Father_Job", "other")),
                                "school": student.get("school", student.get("School", "GP")),
                                "sex": student.get("sex", student.get("Gender", "M")),
                                "address": student.get("address", student.get("Address", "U")),
                                "famsize": student.get("famsize", student.get("Family_Size", "GT3")),
                                "pstatus": student.get("pstatus", student.get("Parental_Status", "T")),
                                "schoolsup": student.get("schoolsup", student.get("School_Support", "no")),
                                "famsup": student.get("famsup", student.get("Family_Support", "yes")),
                                "paid": student.get("paid", student.get("Extra_Paid_Class", "no")),
                                "activities": student.get("activities", student.get("Extra_Curricular_Activities", "yes")),
                                "nursery": student.get("nursery", student.get("Attended_Nursery", "yes")),
                                "higher": student.get("higher", student.get("Wants_Higher_Education", "yes")),
                                "internet": student.get("internet", student.get("Internet_Access", "yes")),
                                "romantic": student.get("romantic", student.get("In_Relationship", "no")),
                            }
                            cleaned_students.append(cleaned)
                        st.write(f"Testing with {len(cleaned_students)} students")
                        st.write("Sample cleaned payload:", cleaned_students[0] if cleaned_students else {})
                        try:
                            r = requests.post(f"{API_URL}/batch_predict", json=cleaned_students, timeout=30)
                            r.raise_for_status()
                            results = r.json()
                            df_sample = df.head(sample_size).copy()
                            df_sample["prediction"] = ["Dropout" if r["prediction"] == 1 else "Active" for r in results]
                            df_sample["probability"] = [f"{r['probability']*100:.1f}%" for r in results]
                            df_sample["confidence"] = [r["confidence"] for r in results]
                            # Afficher uniquement les colonnes demandées
                            columns_to_show = ["school", "sex", "age", "absences", "grade_1", "grade_2", "final_grade", "prediction", "probability", "confidence"]
                            df_display = df_sample[columns_to_show].copy()
                            st.success(f"Predictions completed for {len(results)} students!")
                            st.dataframe(df_display, use_container_width=True)
                            csv = df_display.to_csv(index=False)
                            st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")
                        except Exception as e:
                            st.error(f"Batch prediction error: {e}")
                            if hasattr(e, 'response') and e.response is not None:
                                st.text(f"API response: {e.response.text}")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        st.info("Upload a CSV file with student data.")
