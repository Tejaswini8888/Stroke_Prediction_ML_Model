import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Stroke Prediction ML App",
    page_icon="🧠",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("stroke_pipeline.joblib")

# ---------------- HEADER ----------------
st.title("🧠 Stroke Prediction ML App")
st.subheader("Early risk detection using Machine Learning")

st.markdown(
    """
    **Fast. Reliable. Explainable.**  
    Helping understand stroke risk with data-driven insights.
    """
)

st.markdown("---")

# ---------------- NAVIGATION ----------------
menu = st.radio(
    "Navigation",
    ["Home", "What it does", "Predict", "Instructions"],
    horizontal=True
)

st.markdown("---")

# ---------------- HOME ----------------
if menu == "Home":
    st.header("🏠 Home")
    st.markdown(
        """
        The **Stroke Prediction ML App** is an AI-powered system that predicts
        the **risk of stroke** using health and lifestyle information.

        This project demonstrates **end-to-end Machine Learning deployment**
        using data preprocessing, model training, and a live web interface.
        """
    )
    st.success("👉 Go to **Predict** tab to try the model.")

# ---------------- WHAT IT DOES ----------------
elif menu == "What it does":
    st.header("🩺 What This App Does")
    st.markdown(
        """
        This application uses a trained **Random Forest Machine Learning model**
        to analyze patient data and estimate stroke risk.

        ### 🔍 Key Features
        - Predicts **High Risk / Low Risk** of stroke  
        - Displays **confidence percentage**  
        - Uses medical & lifestyle parameters  
        - Demonstrates explainable ML concepts  

        ### 📊 Inputs Used
        - Age  
        - Hypertension  
        - Heart disease  
        - Average glucose level  
        - BMI  
        - Smoking status  
        - Gender, work type, residence  
        """
    )

# ---------------- PREDICTION ----------------
elif menu == "Predict":
    st.header("🧪 Predict Stroke Risk")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 100, 50)
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
        avg_glucose = st.number_input("Average Glucose Level", 50.0, 300.0, 120.0)

    with col2:
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence = st.selectbox("Residence Type", ["Urban", "Rural"])

    if st.button("🔍 Predict Stroke Risk"):
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
        probability = model.predict_proba(input_df)[0][1] * 100

        st.markdown("---")

        if prediction == 1:
            st.error(f"⚠ **High Risk of Stroke** ({probability:.2f}% confidence)")
            st.markdown(
                """
                **Possible contributing factors:**
                - Higher age
                - Elevated glucose level
                - BMI and lifestyle indicators
                """
            )
        else:
            st.success(f"✅ **Low Risk of Stroke** ({100 - probability:.2f}% confidence)")
            st.markdown(
                """
                **Healthy indicators observed:**
                - Balanced glucose and BMI levels
                - No major risk conditions detected
                """
            )

# ---------------- INSTRUCTIONS ----------------
elif menu == "Instructions":
    st.header("📋 How to Use")
    st.markdown(
        """
        1. Navigate to the **Predict** tab.  
        2. Enter patient health details.  
        3. Click **Predict Stroke Risk**.  
        4. View:
           - Risk result  
           - Confidence score  
           - Explanation of factors  
        """
    )

    st.warning(
        "⚠️ This application is for **educational purposes only** and should not be used for medical diagnosis."
    )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    """
    **Created By — Tejaswini Madarapu**  
    Machine Learning & AI Enthusiast  

    🔗 [GitHub](https://github.com/Tejaswini8888)  
    🔗 [LinkedIn](https://www.linkedin.com/in/tejaswini-madarapu/)
    """
)
