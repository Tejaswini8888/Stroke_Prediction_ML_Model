import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Stroke Risk Predictor",
    page_icon="🧠",
    layout="centered"
)

# ---------------- CUSTOM THEME CSS ----------------
st.markdown("""
<style>
body {
    background-color: #6f63c2;
}

.app-container {
    background: #ffffff;
    padding: 35px;
    border-radius: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    margin-top: 20px;
}

.title {
    text-align: center;
    font-size: 36px;
    font-weight: 700;
    color: #3b2e5a;
}

.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 25px;
}

.disclaimer {
    background: #fdf3d6;
    padding: 18px;
    border-left: 6px solid #d4a017;
    border-radius: 10px;
    font-size: 14px;
    margin-bottom: 25px;
}

.button-primary > button {
    background: linear-gradient(90deg, #6f63c2, #8b7ae6);
    color: white;
    font-size: 18px;
    border-radius: 10px;
    padding: 10px 25px;
    border: none;
}

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

.footer {
    text-align: center;
    margin-top: 40px;
}

.footer a {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    padding: 10px 20px;
    margin: 5px;
    border-radius: 12px;
    color: white;
    text-decoration: none;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("stroke_pipeline.joblib")

# ---------------- APP UI ----------------
st.markdown("<div class='app-container'>", unsafe_allow_html=True)

st.markdown("<div class='title'>🧠 AI Stroke Risk Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Early Risk Detection using Machine Learning</div>", unsafe_allow_html=True)

# ---------------- DISCLAIMER ----------------
st.markdown("""
<div class="disclaimer">
⚠️ <b>IMPORTANT MEDICAL DISCLAIMER</b><br>
This AI tool is for educational purposes only and should NOT replace professional medical advice.
Always consult qualified healthcare professionals for medical decisions.<br><br>
This prediction model may have limitations and should not be used for emergency situations.
If you experience stroke symptoms (sudden numbness, confusion, trouble speaking, severe headache),
seek immediate medical attention by calling emergency services.
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT FORM ----------------
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

st.markdown("<div class='button-primary'>", unsafe_allow_html=True)
predict_btn = st.button("🔍 Analyze Stroke Risk")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if predict_btn:
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

    prediction = model.predict(input_df)[0]

    st.write("")

    if prediction == 1:
        st.markdown("""
        <div class="result-high">
        🚨 <b>High Stroke Risk Detected</b><br>
        Immediate medical consultation is strongly recommended.
        </div>
        """, unsafe_allow_html=True)
        st.progress(85)
    else:
        st.markdown("""
        <div class="result-low">
        ✅ <b>Low Stroke Risk Detected</b><br>
        Maintain a healthy lifestyle and regular checkups.
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
