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
    background: linear-gradient(to right, #eef2f3, #e0eafc);
}
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    color: #1f2933;
}
.subtitle {
    text-align: center;
    color: #555;
    margin-bottom: 30px;
}
.card {
    background: white;
    padding: 28px;
    border-radius: 14px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.disclaimer {
    background: #fff4e5;
    padding: 16px;
    border-left: 6px solid #f59e0b;
    border-radius: 10px;
    font-size: 14px;
}
.result-high {
    background: #fdecea;
    padding: 20px;
    border-left: 6px solid #dc2626;
    border-radius: 12px;
}
.result-low {
    background: #ecfdf3;
    padding: 20px;
    border-left: 6px solid #16a34a;
    border-radius: 12px;
}
.center {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("stroke_pipeline.joblib")

# ---------------- HEADER ----------------
st.markdown("<div class='main-title'>🧠 AI Stroke Risk Predictor</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Early Risk Detection using Machine Learning</div>",
    unsafe_allow_html=True
)

# ---------------- DISCLAIMER ----------------
st.markdown("""
<div class="disclaimer">
⚠️ <b>Medical Disclaimer:</b><br>
This tool is for educational purposes only and does not replace professional medical advice.
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- LAYOUT ----------------
left, right = st.columns([2, 1])

# ---------------- INPUT FORM ----------------
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🩺 Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 1, 100, 45)
        hypertension = st.radio("Hypertension", [0, 1], horizontal=True)
        heart_disease = st.radio("Heart Disease", [0, 1], horizontal=True)

    with col2:
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
        work_type = st.selectbox(
            "Work Type",
            ["Private", "Self-employed", "Govt_job", "children"]
        )
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        smoking = st.selectbox(
            "Smoking Status",
            ["never smoked", "formerly smoked", "smokes"]
        )

    glucose = st.number_input(
        "Average Glucose Level (mg/dL)", 50.0, 300.0, 110.0
    )
    bmi = st.number_input("BMI", 10.0, 60.0, 26.0)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Analyze Stroke Risk", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- INFO CARD ----------------
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("ℹ️ About This App")
    st.markdown("""
- Uses trained **Machine Learning model**
- Combines medical & lifestyle data
- Provides quick **risk classification**
- Designed for **early awareness**
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if predict_btn:
    with st.spinner("Analyzing patient data..."):
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
        Preventive medical consultation is recommended.
        </div>
        """, unsafe_allow_html=True)
        st.progress(80)
    else:
        st.markdown("""
        <div class="result-low">
        ✅ <b>Low Stroke Risk Detected</b><br>
        Maintain a healthy lifestyle and regular checkups.
        </div>
        """, unsafe_allow_html=True)
        st.progress(25)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<div class='center'>"
    "Created by <b>Tejaswini Madarapu</b> • "
    "<a href='https://github.com/Tejaswini8888'>GitHub</a> • "
    "<a href='https://www.linkedin.com/in/tejaswini-madarapu/'>LinkedIn</a>"
    "</div>",
    unsafe_allow_html=True
)
