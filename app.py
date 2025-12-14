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
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #6a11cb, #2575fc);
}
.block-container {
    padding-top: 2rem;
}
.card {
    background: #ffffff;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.12);
    margin-bottom: 20px;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: white;
}
.subtitle {
    text-align: center;
    color: #e0e0e0;
    margin-bottom: 25px;
}
.disclaimer {
    background: #fff3cd;
    padding: 15px;
    border-left: 6px solid #ff9800;
    border-radius: 10px;
    font-size: 14px;
}
.cta {
    background: linear-gradient(to right, #2575fc, #6a11cb);
    color: white;
    padding: 12px;
    border-radius: 10px;
    font-weight: bold;
    text-align: center;
}
.high-risk {
    background: #fdecea;
    border-left: 6px solid #dc3545;
    padding: 18px;
    border-radius: 10px;
}
.low-risk {
    background: #e7f5ec;
    border-left: 6px solid #28a745;
    padding: 18px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("stroke_pipeline.joblib")

# ---------------- HEADER ----------------
st.markdown("<div class='title'>🧠 AI Stroke Risk Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Advanced Machine Learning for Healthcare Screening</div>", unsafe_allow_html=True)

# ---------------- DISCLAIMER ----------------
st.markdown("""
<div class="disclaimer">
⚠️ <b>Medical Disclaimer:</b>  
This tool is for educational purposes only and does NOT replace professional medical advice.
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- MAIN LAYOUT ----------------
left, right = st.columns([2.2, 1])

# ---------------- LEFT: PATIENT FORM ----------------
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🩺 Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 1, 100, 45)
        hypertension = st.selectbox("Hypertension", [0, 1])
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
        residence = st.selectbox("Residence Type", ["Urban", "Rural"])

    with col2:
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children"])
        glucose = st.number_input("Glucose Level (mg/dL)", 50.0, 300.0, 110.0)
        bmi = st.number_input("BMI (optional)", 10.0, 60.0, 26.0)
        smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])

    analyze = st.button("🔍 Analyze Stroke Risk")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RIGHT: VISUAL PANEL ----------------
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧠 Ready for Analysis")
    st.image(
        "https://images.unsplash.com/photo-1581090464777-f3220bbe1b8b",
        use_container_width=True
    )
    st.markdown("""
    AI analyzes patient health indicators and estimates stroke risk using
    trained machine learning models.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if analyze:
    input_df = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "residence_type": residence,
        "avg_glucose_level": glucose,
        "bmi": bmi,
        "smoking_status": smoking
    }])

    pred = model.predict(input_df)[0]

    st.write("")
    if pred == 1:
        st.markdown("""
        <div class="high-risk">
        🚨 <b>High Stroke Risk Detected</b><br>
        Immediate medical consultation is recommended.
        </div>
        """, unsafe_allow_html=True)
        st.progress(90)
    else:
        st.markdown("""
        <div class="low-risk">
        ✅ <b>Low Stroke Risk Detected</b><br>
        Maintain a healthy lifestyle and regular checkups.
        </div>
        """, unsafe_allow_html=True)
        st.progress(30)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:white;'>"
    "Created by <b>Tejaswini Madarapu</b> | "
    "<a href='https://github.com/Tejaswini8888' style='color:white;'>GitHub</a> | "
    "<a href='https://www.linkedin.com/in/tejaswini-madarapu/' style='color:white;'>LinkedIn</a>"
    "</div>",
    unsafe_allow_html=True
)
