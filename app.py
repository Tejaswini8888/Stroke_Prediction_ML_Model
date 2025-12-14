import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Stroke Risk Predictor",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #664C36, #331C08);
        color: #ffffff;
    }

    /* =====================================================
       ✅ NORMALIZE ALL LABEL STRUCTURES (Case 1 + Case 2)
       Case 1: <label>Age</label>
       Case 2: <label><p>Gender</p></label>
       ===================================================== */
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] label p {
        color: #ffffff !important;
        font-weight: 600 !important;
        opacity: 1 !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.2 !important;
    }

    /* Input text (unchanged) */
    .stSelectbox div,
    .stNumberInput input {
        color: #2b1a0f !important;
    }

    /* Titles */
    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 16px;
        opacity: 0.9;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 15px;
        color: #ffffff;
    }

    .disclaimer {
        background: rgba(255, 255, 255, 0.15);
        padding: 18px;
        border-left: 6px solid #f5c542;
        border-radius: 10px;
        font-size: 14px;
        margin-bottom: 25px;
    }

    .stButton > button {
        background: #331C08;
        color: white;
        border-radius: 12px;
        padding: 12px 20px;
        font-size: 16px;
        font-weight: 600;
        border: none;
    }

    .stButton > button:hover {
        background: #4a2a12;
    }

    .result-high {
        background: rgba(220, 53, 69, 0.15);
        padding: 20px;
        border-left: 6px solid #dc3545;
        border-radius: 12px;
    }

    .result-low {
        background: rgba(40, 167, 69, 0.15);
        padding: 20px;
        border-left: 6px solid #28a745;
        border-radius: 12px;
    }

    .footer {
        text-align: center;
        margin-top: 40px;
        opacity: 0.9;
    }

    .footer a {
        display: inline-block;
        margin: 10px;
        padding: 12px 22px;
        background: rgba(255,255,255,0.18);
        color: white;
        border-radius: 14px;
        text-decoration: none;
        font-weight: 600;
    }

    .footer a:hover {
        background: rgba(255,255,255,0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("stroke_pipeline.joblib")

# ---------------- HEADER ----------------
st.markdown("<div class='main-title'>🧠 AI Stroke Risk Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Early Stroke Risk Detection using Machine Learning</div>", unsafe_allow_html=True)

# ---------------- DISCLAIMER ----------------
st.markdown("""
<div class="disclaimer">
⚠️ <b>IMPORTANT MEDICAL DISCLAIMER</b><br>
This AI tool is for educational purposes only and should NOT replace professional medical advice.
Always consult qualified healthcare professionals for medical decisions.
If you experience symptoms such as sudden numbness, confusion, trouble speaking, or severe headache,
seek immediate medical attention.
</div>
""", unsafe_allow_html=True)

# ---------------- PATIENT INFO ----------------
st.markdown("<div class='section-title'>🩺 Patient Information</div>", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 1, 100, 45)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])

with c2:
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    glucose = st.number_input("Avg Glucose Level (mg/dL)", 50.0, 300.0, 110.0)
    bmi = st.number_input("BMI", 10.0, 60.0, 26.0)
    smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])

analyze = st.button("🔍 Analyze Stroke Risk")

# ---------------- PREDICTION ----------------
if analyze:
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

    pred = model.predict(input_df)[0]

    if pred == 1:
        st.markdown("""
        <div class="result-high">
        🚨 <b>High Stroke Risk Detected</b><br>
        Please consult a medical professional immediately.
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
st.markdown("""
<div class="footer">
🔒 Powered by Advanced Machine Learning • Built with ❤️ for Healthcare<br><br>
<a href="https://github.com/Tejaswini8888" target="_blank">👩‍💻 GitHub</a>
<a href="https://www.linkedin.com/in/tejaswini-madarapu/" target="_blank">💼 LinkedIn</a>
</div>
""", unsafe_allow_html=True)
