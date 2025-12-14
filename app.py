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
    background-color: #f4f6fb;
}
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
}
.subtitle {
    text-align: center;
    color: #555;
    margin-bottom: 30px;
}
.card {
    background: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.disclaimer {
    background: #fff3cd;
    padding: 18px;
    border-left: 6px solid #ffc107;
    border-radius: 8px;
    font-size: 14px;
}
.result-high {
    background: #fdecea;
    padding: 20px;
    border-left: 6px solid #dc3545;
    border-radius: 10px;
}
.result-low {
    background: #e7f5ec;
    padding: 20px;
    border-left: 6px solid #28a745;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("stroke_pipeline.joblib")

# ---------------- HEADER ----------------
st.markdown("<div class='main-title'>🧠 AI Stroke Risk Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Advanced Machine Learning for Healthcare Screening</div>", unsafe_allow_html=True)

# ---------------- DISCLAIMER ----------------
st.markdown("""
<div class="disclaimer">
⚠️ <b>IMPORTANT MEDICAL DISCLAIMER</b><br>
This AI tool is for educational purposes only and should NOT replace professional medical advice.
Always consult qualified healthcare professionals for medical decisions.
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- MAIN LAYOUT ----------------
left, right = st.columns([2,1])

# ---------------- INPUT CARD ----------------
with left:
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

# ---------------- SIDE VISUAL CARD ----------------
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧠 Ready for Analysis")
    st.image(
        "https://images.unsplash.com/photo-1581091012184-5c7b6c5b3a63",
        use_container_width=True
    )
    st.markdown("AI evaluates risk based on health indicators.")
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
st.markdown("---")
st.markdown(
    "Created by **Tejaswini Madarapu** | "
    "[GitHub](https://github.com/Tejaswini8888) | "
    "[LinkedIn](https://www.linkedin.com/in/tejaswini-madarapu/)"
)
