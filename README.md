# 🧠 Stroke Prediction ML Model

This project predicts whether a person is at risk of **stroke (1)** or **not (0)** using machine learning.  
It uses medical and lifestyle data such as age, hypertension, heart disease, glucose level, BMI, smoking status, etc.

The model is built using a complete ML pipeline

(preprocessing → training → evaluation → prediction).

---

## 📂 Dataset

The dataset **stroke.csv** contains the following columns:

age, hypertension, heart_disease, ever_married, avg_glucose_level,
bmi, smoking_status, gender, work_type, residence_type, stroke

yaml
Copy code

- `stroke` is the target column (0 = No Stroke, 1 = Stroke).

This dataset is synthetic and created for learning purposes.

---

## 🔧 Requirements

Install all dependencies:

pip install -r requirements.txt


Your `requirements.txt` should contain:

pandas\
numpy\
scikit-learn\
matplotlib\
seaborn


> **Note:**  
> If using scikit-learn 1.2 or above, the encoder should use:  
> `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`

---

## ▶️ How to Run
Run the script:
```
python main.py
```

This will:

1. Load the dataset  
2. Preprocess numeric and categorical features  
3. Train a RandomForest model  
4. Evaluate model performance  
5. Display a confusion matrix  
6. Predict stroke risk for a sample new patient  

---

## 📊 Model Evaluation

The script prints:

- **Accuracy**
- **Classification Report** (precision, recall, F1-score)
- **Confusion Matrix** (visualized using Seaborn)

These help you understand how well the model performs, especially on medical predictions where recall is important.

---

## 🧪 Example Prediction

The program includes an example input for a new patient and prints:

1 = High risk of stroke
0 = No stroke risk

---

## 🚀 Live Demo
🔗 https://strokepredictionmlmodel.streamlit.app

This web app predicts the risk of stroke based on patient health data using a trained Random Forest model and preprocessing pipeline.


## 🚀 Future Improvements

You can enhance this project by adding:

- Class imbalance handling (SMOTE, class weights)
- Hyperparameter tuning (GridSearchCV)
- Feature importance analysis
- Model explainability (SHAP)
- XGBoost or LightGBM models

---

## 👩‍💻 Author

**Tejaswini Madarapu**  
- GitHub: https://github.com/Tejaswini8888  
- LinkedIn: https://www.linkedin.com/in/tejaswini-madarapu/

---

## 📄 License

This project is open for learning and portfolio use.
