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

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("🧠 Stroke AI")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "⚙️ What it Does", "🧪 Try Demo", "📋 Instructions"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Created by**  
    Tejaswini Madarapu  

    🔗 [GitHub](https://github.com/Tejaswini8888)  
    🔗 [LinkedIn](https://www.linkedin.com/in/tejaswini-madarapu/)
    """
)

# ---------------- HOME (HERO SECTION) ----------------
if page == "🏠 Home":
    st.markdown(
        """
        <h1 style='text-align:center;'>🧠 Stroke Prediction AI</h1>
        <h3 style='text-align:center; color:gray;'>
        Early risk detection using Machine Learning
        </h3>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style='text-align:center; font-size:18px;'>
        Fast. Fair. Explainable.  
        Helping understand stroke risk with data-driven insights.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🚀 Get Started"):
            st.session_state.page = "🧪 Try Demo"
            st.experimental_rerun()

    st.markdown("---")

    st.subheader("Why this project?")
    st.markdown(
        """
        - Demonstrates **end-to-end Machine Learning**
        - Uses **real healthcare data**
        - Includes **model explainability**
        - Fully **deployed as a web application**
        """
    )

# ---------------- WHAT IT DOES ----------------
elif page == "⚙️ What it Does":
    st.header("⚙️ What This App Does")

    st.markdown(
        """
        The **Stroke Prediction AI App** uses a trained **Machine Learning model**
        to predict whether a person is at risk of stroke based on medical and
        lifestyle factors.

        ### 🔍 Key Capabilities
        - Predicts **High / Low stroke risk**
        - Displays **confidence score**
        - Uses preprocessing + ML pipeline
        - Designed for **educational & demo purposes**

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

# ---------------- TRY DEMO (PREDICTION) ----------------
elif page == "🧪 Try Demo":
    st.header("🧪 Try the Stroke Prediction Demo")

    # Demo buttons
    colA, colB = st.columns(2)
    with colA:
        demo_high = st.button("🧪 Load High-Risk Example")
    with colB:
        demo_low = st.button("🧪 Load Low-Risk Example")

    # Default values
    age, hypertension, heart_disease = 50, 0, 0
    ever_married, avg_glucose, bmi = "Yes", 120.0, 25.0
    smoking, gender, work_type, residence = "never smoked", "Male", "Private", "Urban"

    if demo_high:
        age, hypertension, avg_glucose, bmi, smoking = 70, 1, 180.0, 33.0, "formerly smoked"

    if demo_low:
        age, hypertension, avg_glucose, bmi, smoking = 35, 0, 95.0, 22.0, "never smoked"

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 100, age)
        hypertension = st.selectbox("Hypertension", [0,1], index=hypertension)
        heart_disease = st.selectbox("Heart Disease", [0,1], index=heart_disease)
        ever_married = st.selectbox("Ever Married", ["Yes","No"], index=0)
        avg_glucose = st.number_input("Average Glucose Level", 50.0, 300.0, avg_glucose)

    with col2:
        bmi = st.number_input("BMI", 10.0, 60.0, bmi)
        smoking = st.selectbox("Smoking Status",
                               ["never smoked","formerly smoked","smokes","Unknown"],
                               index=["never smoked","formerly smoked","smokes","Unknown"].index(smoking))
        gender = st.selectbox("Gender", ["Male","Female","Other"])
        work_type = st.selectbox("Work Type",
                                 ["Private","Self-employed","Govt_job","children","Never_worked"])
        residence = st.selectbox("Residence Type", ["Urban","Rural"])

    if st.button("🔍 Predict Stroke Risk"):
        input_df = pd.DataFrame({
            "age":[age],
            "hypertension":[hypertension],
            "heart_disease":[heart_disease],
            "ever_married":[ever_married],
            "avg_glucose_level":[avg_glucose],
            "bmi":[bmi],
            "smoking_status":[smoking],
            "gender":[gender],
            "work_type":[work_type],
            "residence_type":[residence]
        })

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] * 100

        st.markdown("---")
        with st.container():
            if pred == 1:
                st.error(f"⚠ **High Stroke Risk** — {prob:.2f}% confidence")
                st.markdown(
                    """
                    **Likely contributing factors:**
                    - Higher age
                    - Elevated glucose level
                    - BMI & lifestyle indicators
                    """
                )
            else:
                st.success(f"✅ **Low Stroke Risk** — {100-prob:.2f}% confidence")
                st.markdown(
                    """
                    **Positive indicators observed:**
                    - Balanced glucose & BMI
                    - No major risk conditions detected
                    """
                )

# ---------------- INSTRUCTIONS ----------------
elif page == "📋 Instructions":
    st.header("📋 How to Use")

    st.markdown(
        """
        1. Navigate to **Try Demo**  
        2. Enter health details or load demo data  
        3. Click **Predict Stroke Risk**  
        4. View result, confidence, and explanation  
        """
    )

    st.warning(
        "⚠️ This application is for **educational purposes only** and not a medical diagnosis tool."
    )
