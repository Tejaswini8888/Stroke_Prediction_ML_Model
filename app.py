import streamlit as st
import pandas as pd
import joblib

st.title("🧠 Stroke Prediction App")

st.write("Enter patient details to predict stroke risk")

# Load trained pipeline
model = joblib.load("stroke_pipeline.joblib")

# Inputs
age = st.number_input("Age", 1, 100, 50)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
avg_glucose = st.number_input("Average Glucose Level", 50.0, 300.0, 120.0)
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence = st.selectbox("Residence Type", ["Urban", "Rural"])

if st.button("Predict"):
    input_df = pd.DataFrame({
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "ever_married": [ever_married],
        "avg_glucose_level": [avg_glucose],
        "bmi": [bmi],
        "smoking_status": [smoking],
        "gender": [gender],
        "work_type": [work_type],
        "residence_type": [residence]
    })

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠ High Risk of Stroke")
    else:
        st.success("✅ Low Risk of Stroke")
