import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Stroke Risk Predictor",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM CSS (DARK UI LIKE IMAGE) ----------------
st.markdown("""
<style>

/* App background */
.stApp {
    background: radial-gradient(circle at top, #14171a, #0b0d10);
    color: #ffffff;
}

/* Center container */
.block-container {
    max-width: 900px;
    padding-top: 2.5rem;
}

/* Title */
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 6px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 15px;
    opacity: 0.85;
    margin-bottom: 35px;
}

/* Section title */
.section-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 18px;
}

/* ALL LABELS — EXACT FIX */
div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] p {
    color: #ffffff !important;
    font-weight: 500 !important;
    opacity: 1 !important;
}

/* Inputs */
.stSelectbox div,
.stNumberInput input,
.stTextInput input {
    background-color: #1e2228 !important;
    color: #ffffff !important;
    border-radius: 8px;
}

/* Buttons */
.stButton > button {
    background: #2a2f36;
    color: white;
    border-radius: 10px;
    padding: 10px 22px;
    font-size: 15px;
    font-weight: 600;
    border: none;
}
.stButton > button:hover {
    background: #3a4048;
}

/* Result cards */
.result-high {
    background: rgba(220, 53, 69, 0.15);
    padding: 18px;
    border-left: 5px solid #dc3545;
    border-radius: 10px;
}
.result-low {
    background: rgba(40, 167, 69, 0.15);
    padding: 18px;
    border-left: 5px solid #28a745;
    border-radius: 10px;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 50px;
    font-size: 13px;
    opacity: 0.7;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("stroke_pipeline.joblib")

# ---------------- HEADER ----------------
st.markdown("<div class='main-title'>🧠 AI Stroke Risk Predictor</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Early Stroke Risk Detection using Machine Learning</div>",
    unsafe_allow_html=True
)

# ---------------- INPUT FORM ----------------
st.markdown("<div class='section-title'>Enter Your Health Information</div>", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 1, 100, 25)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

with c2:
    glucose = st.number_input("Average Glucose Level (mg/dL)", 50.0, 300.0, 91.05)
    bmi = st.number_input("BMI", 10.0, 60.0, 23.46)
    smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])

# Convert Yes/No → 0/1
hypertension = 1 if hypertension == "Yes" else 0
heart_disease = 1 if heart_disease == "Yes" else 0

predict = st.button("Predict Stroke Risk")

# ---------------- PREDICTION ----------------
if predict:
    input_df = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": "Yes",
        "work_type": "Private",
        "residence_type": "Urban",
        "avg_glucose_level": glucose,
        "bmi": bmi,
        "smoking_status": smoking
    }])

    pred = model.predict(input_df)[0]

    if pred == 1:
        st.markdown("""
        <div class="result-high">
        🚨 <b>High Stroke Risk Detected</b><br>
        Please consult a healthcare professional.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-low">
        ✅ <b>Low Stroke Risk Detected</b><br>
        Maintain a healthy lifestyle.
        </div>
        """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
© 2025 AI Stroke Risk Predictor • Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
