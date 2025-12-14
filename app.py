import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Stroke Prediction AI",
    page_icon="🧠",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("stroke_pipeline.joblib")

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 Stroke AI")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "⚙️ What it Does", "🧪 Try Demo", "📋 Instructions"],
    index=["🏠 Home", "⚙️ What it Does", "🧪 Try Demo", "📋 Instructions"].index(st.session_state.page)
)

st.session_state.page = page

st.sidebar.markdown("---")
st.sidebar.markdown("**Created by**")
st.sidebar.markdown("**Tejaswini Madarapu**")
st.sidebar.markdown("[GitHub](https://github.com/Tejaswini8888)")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/tejaswini-madarapu/)")

# ---------------- HOME ----------------
if page == "🏠 Home":
    st.markdown("<h1 style='text-align:center;'>Early Stroke Risk Detection</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;'>Fast • Fair • Explainable AI for healthcare decisions</p>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🚀 Get Started"):
            st.session_state.page = "🧪 Try Demo"
            st.rerun()

# ---------------- WHAT IT DOES ----------------
elif page == "⚙️ What it Does":
    st.header("⚙️ What This App Does")

    st.markdown("""
    This AI-powered application predicts **stroke risk** using patient health data.

    **Model used:** Random Forest Classifier  
    **Preprocessing:** Scaling + One-Hot Encoding  
    **Deployment:** Streamlit Cloud

    ### Why this matters:
    - Helps in early detection
    - Saves medical time
    - Supports data-driven decisions
    """)

# ---------------- TRY DEMO ----------------
elif page == "🧪 Try Demo":
    st.header("🧪 Stroke Risk Prediction Demo")

    st.markdown("Enter patient details or try demo profiles.")

    # ---- Demo buttons ----
    colA, colB = st.columns(2)
    with colA:
        if st.button("🧪 Load High Risk Patient"):
            demo = {
                "age": 72, "hypertension": 1, "heart_disease": 1,
                "ever_married": "Yes", "avg_glucose_level": 180.0,
                "bmi": 34.5, "smoking_status": "formerly smoked",
                "gender": "Male", "work_type": "Private", "residence_type": "Urban"
            }
    with colB:
        if st.button("🧪 Load Low Risk Patient"):
            demo = {
                "age": 30, "hypertension": 0, "heart_disease": 0,
                "ever_married": "No", "avg_glucose_level": 90.0,
                "bmi": 22.0, "smoking_status": "never smoked",
                "gender": "Female", "work_type": "Private", "residence_type": "Urban"
            }

    if "demo" not in locals():
        demo = {
            "age": 45, "hypertension": 0, "heart_disease": 0,
            "ever_married": "Yes", "avg_glucose_level": 110.0,
            "bmi": 26.0, "smoking_status": "never smoked",
            "gender": "Male", "work_type": "Private", "residence_type": "Urban"
        }

    # ---- Input form ----
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 1, 100, demo["age"])
            glucose = st.number_input("Avg Glucose Level", value=demo["avg_glucose_level"])
            bmi = st.number_input("BMI", value=demo["bmi"])
            married = st.selectbox("Ever Married", ["Yes", "No"], index=["Yes","No"].index(demo["ever_married"]))

        with col2:
            hypertension = st.selectbox("Hypertension", [0,1], index=demo["hypertension"])
            heart_disease = st.selectbox("Heart Disease", [0,1], index=demo["heart_disease"])
            smoking = st.selectbox("Smoking Status", ["never smoked","formerly smoked","smokes"],
                                   index=["never smoked","formerly smoked","smokes"].index(demo["smoking_status"]))
            gender = st.selectbox("Gender", ["Male","Female"], index=["Male","Female"].index(demo["gender"]))

        work_type = st.selectbox("Work Type", ["Private","Self-employed","Govt_job","children"])
        residence = st.selectbox("Residence Type", ["Urban","Rural"])

        submit = st.form_submit_button("🔍 Predict Risk")

    if submit:
        input_df = pd.DataFrame([{
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "ever_married": married,
            "avg_glucose_level": glucose,
            "bmi": bmi,
            "smoking_status": smoking,
            "gender": gender,
            "work_type": work_type,
            "residence_type": residence
        }])

        prediction = model.predict(input_df)[0]

        st.markdown("---")
        if prediction == 1:
            st.error("🚨 **High Stroke Risk Detected**")
            st.progress(85)
            st.markdown("**Recommendation:** Immediate medical consultation advised.")
        else:
            st.success("✅ **Low Stroke Risk Detected**")
            st.progress(25)
            st.markdown("**Recommendation:** Maintain healthy lifestyle.")

# ---------------- INSTRUCTIONS ----------------
elif page == "📋 Instructions":
    st.header("📋 How to Use")
    st.markdown("""
    1. Navigate to **Try Demo**
    2. Enter patient details or load demo
    3. Click **Predict Risk**
    4. View result & recommendation
    """)

