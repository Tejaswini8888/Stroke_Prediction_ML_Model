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

/* FULL APP BACKGROUND */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #664C36, #331C08);
}

/* REMOVE STREAMLIT HEADER SPACE */
[data-testid="stHeader"] {
    background: transparent;
}

/* MAIN CONTENT CARD */
.main-card {
    background: #ffffff;
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.15);
    margin-top: 20px;
}

/* HEADER BOX */
.header-box {
    background: #ffffff;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 25px;
}

.header-box h1 {
    color: #331C08;
    font-size: 36px;
    font-weight: 800;
}

/* DISCLAIMER */
.disclaimer {
    background: #fff3cd;
    padding: 16px;
    border-left: 6px solid #ffc107;
    border-radius: 12px;
    font-size: 14px;
}

/* RESULTS */
.result-high {
    background: #fdecea;
    padding: 18px;
    border-left: 6px solid #dc3545;
    border-radius: 12px;
}

.result-low {
    background: #e7f5ec;
    padding: 18px;
    border-left: 6px solid #28a745;
    border-radius: 12px;
}

/* BUTTON STYLE */
.stButton > button {
    background: linear-gradient(135deg, #664C36, #331C08);
    color: white;
    font-size: 16px;
    padding: 12px 22px;
    border-radius: 14px;
    border: none;
}

/* FOOTER BUTTONS */
.footer-btn {
    display: inline-block;
    padding: 12px 22px;
    margin: 10px;
    background: linear-gradient(135deg, #664C36, #331C08);
    color: white !important;
    border-radius: 14px;
    text-decoration: none;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
model = joblib.load("stroke_pipeline.joblib")

# ---------------- HEADER ----------------
st.markdown("""
<div class="header-box">
    <h1>🧠 AI Stroke Risk Predictor</h1>
    <p>Early Risk Detection using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ---------------- DISCLAIMER ----------------
st.markdown("""
<div class="disclaimer">
⚠️ <b>IMPORTANT MEDICAL DISCLAIMER</b><br>
This AI tool is for educational purposes only and should NOT replace professional medical advice.
Always consult qualified healthcare professionals for medical decisions.<br><br>
If you experience stroke symptoms (sudden numbness, confusion, trouble speaking, severe headache),
seek immediate medical attention by calling emergency services.
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- INPUT FORM ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🩺 Patient Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 1, 100, 45)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])

with col2:
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    glucose = st.number_input("Avg Glucose Level (mg/dL)", 50.0, 300.0, 110.0)
    bmi = st.number_input("BMI", 10.0, 60.0, 26.0)
    smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])

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
        Immediate medical consultation is recommended.
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

# ---------------- FOOTER ----------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;">
    <p>🔒 Powered by Advanced Machine Learning • Built with ❤️ for Healthcare</p>
    <a class="footer-btn" href="https://github.com/Tejaswini8888" target="_blank">🐙 GitHub</a>
    <a class="footer-btn" href="https://www.linkedin.com/in/tejaswini-madarapu/" target="_blank">💼 LinkedIn</a>
</div>
""", unsafe_allow_html=True)
