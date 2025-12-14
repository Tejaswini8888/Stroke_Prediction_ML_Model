# 🧠 AI Stroke Risk Prediction System

An **AI-powered Stroke Risk Prediction System** built using **Python, Machine Learning, and Streamlit**.  
This application helps in **early stroke risk assessment** by analyzing patient health data using a trained ML model, presented through a clean and interactive healthcare-focused UI.

---

## 🌐 Live Demo

🔗 **Try the App Here:**  
https://strokepredictionmlmodel-p.streamlit.app/

---

## 🚀 Features

- 🧠 AI-based stroke risk prediction
- 🩺 User-friendly patient information form
- 🎯 Explicit dropdown selections (Select-first UX)
- 📊 Predicts **High Risk** or **Low Risk**
- ⚠️ Medical disclaimer for ethical AI usage
- ✨ Interactive UI with modern teal theme
- ⚡ Fast predictions using a pre-trained ML pipeline

---

## 🧠 Prediction Approach

- A **supervised machine learning model** trained on healthcare data
- Uses patient attributes such as:
  - Age
  - Gender
  - Hypertension
  - Heart disease
  - BMI
  - Average glucose level
  - Smoking status
  - Work type
  - Residence type
- Model is loaded using a **saved pipeline (`joblib`)**
- Predicts stroke risk as:
  - ✅ Low Risk
  - 🚨 High Risk

---

## 🛠️ Tech Stack

- **Language:** Python
- **Web Framework:** Streamlit
- **Machine Learning:** Scikit-learn
- **Model Handling:** Joblib
- **Data Processing:** Pandas
- **Styling:** Custom CSS (Healthcare UI)

---

## 📂 Project Structure

```
├── app.py                     # Main Streamlit application
├── stroke_pipeline.joblib     # Trained ML pipeline
├── requirements.txt           # Python dependencies
├── .runtime.txt               # Python version for Streamlit Cloud
├── README.md                  # Project documentation
└── .streamlit/
    └── secrets.toml           # (If required for future extensions)
```

---

## ▶️ Run Locally

```bash
# Clone the repository
git clone https://github.com/Tejaswini8888/Stroke_Prediction_ML_Model.git

# Navigate into the project
cd Stroke_Prediction_ML_Model

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## ⚠️ Medical Disclaimer

This application is intended **for educational and demonstration purposes only**.  
It should **NOT** be used as a substitute for professional medical diagnosis or treatment.  
Always consult qualified healthcare professionals for medical advice.

---

## 👩‍💻 Author

**Tejaswini Madarapu**

- GitHub: https://github.com/Tejaswini8888  
- LinkedIn: https://www.linkedin.com/in/tejaswini-madarapu/

---

## ⭐ Acknowledgements

- Public healthcare datasets used for training
- Streamlit for the web application framework
- Scikit-learn for machine learning utilities

---

## 📜 License

This project is licensed under the **MIT License**.

---

✨ **If you find this project useful, don’t forget to give it a ⭐ on GitHub!**
