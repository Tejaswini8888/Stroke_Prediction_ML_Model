import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Stroke Risk Predictor",
    page_icon="🧠",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* Full background */
.stApp {
    background: linear-gradient(135deg, #6f63c2, #8f7ae6);
}

/* Main card */
.main-card {
    background: white;
    padding: 35px;
    border-radius: 20px;
    max-width: 850px;
    margin: auto;
    box-shadow: 0 15px 40px rgba(0,0,0,0.15);
}

/* Titles */
.title {
    text-align: center;
    font-size: 36px;
    font-weight: 700;
    color: #3b2e5a;
}

.subtitle {
    text-align: center;
    color: #555;
    margin-bottom: 25px;
}

/* Disclaimer */
.disclaimer {
    background: #fff3cd;
    padding: 18px;
    border-left: 6px solid #f0ad4e;
    border-radius: 10px;
    font-size: 14px;
    margin-bottom: 25px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #6f63c2, #8f7ae6);
    color: white;
    font-size: 18px;
    border-radius: 12px;
    padding: 12px;
    border: none;
}

/* Result boxes */
.result-high {
    background: #fdecea;
    padding: 20px;
    border-left: 6px solid #dc3545;
    border-radius: 12px;
    font-size: 18px;
}

.result-low {
    background: #e7f5ec;
    padding: 20px;
    border-left: 6px solid #2e8b57;
    border-radius: 12px;
    font-size: 18px;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 35px;
}

.footer a {
    display: inline-block;
    background: rgba(255,255,255,0.25);
    color: white;
    padding: 12px 22px;
    margin: 6px;
    border-radius: 16px;
    font-weight: 600;
    text-decoration: none;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("stroke_pipeline.joblib")

# ---------------- UI START ----------------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

st.markdown("<div class='title'>🧠 AI Stroke Risk Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Early Risk Detection using Machine Learning</div>", unsafe_allow_html=True)

# ---------------- DISCLAIMER ----------------
st.markdown("""
<div class="disclaimer">
⚠️ <b>IMPORTANT MEDICAL DISCLAIMER</b><br>
This AI tool is for educational purposes only and should NOT replace professional medical advice.
Always consult qualified healthcare professionals.<br><br>
This model should not be used in emergency situations.
If you experience stroke symptoms (sudden numbness, confusion, trouble speaking, severe headache),
seek immediate medical help.
</div>
""", unsafe_allow_html=True)

# ---------------- FORM ----------------
st.subheader("🩺 Patient Information")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 1, 100, 45)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
glucose = st.number_input("Avg Glucose Level (mg/dL)", 50.0, 300.0, 110.0)
bmi = st.number_input("BMI", 10.0, 60.0, 26.0)
smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])

predict = st.button("🔍 Analyze Stroke Risk")

# ---------------- PREDICTION ----------------
if predict:
    input_df = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "residence_type": residence_type,
        "avg_glucose_level": glucose,
        "bmi": bmi,
        "smoking_status": smoking
    }])

    result = model.predict(input_df)[0]

    st.write("")

    if result == 1:
        st.markdown("""
        <div class="result-high">
        🚨 <b>High Stroke Risk Detected</b><br>
        Please consult a healthcare professional immediately.
        </div>
        """, unsafe_allow_html=True)
        st.progress(85)
    else:
        st.markdown("""
        <div class="result-low">
        ✅ <b>Low Stroke Risk Detected</b><br>
        Continue maintaining a healthy lifestyle.
        </div>
        """, unsafe_allow_html=True)
        st.progress(25)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER BUTTONS ----------------
st.markdown("""
<div class="footer">
    <a href="https://github.com/Tejaswini8888" target="_blank">💻 GitHub</a>
    <a href="https://www.linkedin.com/in/tejaswini-madarapu/" target="_blank">🔗 LinkedIn</a>
</div>
""", unsafe_allow_html=True)
