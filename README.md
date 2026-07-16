# Disease Prediction System

[![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7+-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.3+-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.47+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Joblib](https://img.shields.io/badge/Joblib-Model%20Persistence-success)](https://joblib.readthedocs.io)
[![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-green)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

A Machine Learning-based Disease Prediction System that predicts diseases from patient symptoms using a Random Forest Classifier.

---

## Project Structure

```
Disease_Prediction_Model/
│
├── data/
│   ├── archive.zip                     ← Raw disease dataset
│   └── Final_Disease_dataset.csv       ← Extracted dataset
│
├── model/
│   ├── evaluation_report.txt           ← Model evaluation metrics
│   ├── full_model.pkl                  ← Complete RandomForestModel object
│   ├── label.pkl                       ← Saved LabelEncoder
│   └── predict_model.pkl               ← Trained Random Forest model
│
├── src/                                ← Core machine learning modules
│   ├── data_loader.py                  ← Dataset extraction and loading
│   ├── data_splitter.py                ← Train/validation/test split
│   ├── model_creation.py               ← Random Forest training and prediction
│   ├── model_evaluator.py              ← Model evaluation metrics
│   └── preprocessing.py                ← Data preprocessing and label encoding
│
├── app.py                              ← Streamlit web application
├── Main.py                             ← Training and evaluation entry point
├── predict.py                          ← Loads trained model and performs prediction
│
├── README.md                           ← Project documentation
├── requirements.txt                    ← Project dependencies
└── .gitignore                          ← Git ignore rules
```

## Project Workflow

```
Dataset
   │
   ▼
Data Loading
   │
   ▼
Data Preprocessing
   │
   ├── Remove Duplicate Rows
   ├── Remove Rare Diseases
   ├── Reduce Dataset Size
   ├── Remove Constant Features
   └── Encode Disease Labels
   │
   ▼
Data Splitting
   │
   ├── Training Set (70%)
   ├── Validation Set (10%)
   └── Test Set (20%)
   │
   ▼
Random Forest Training
   │
   ├── Hyperparameter Optimization
   ├── Cross Validation
   └── Model Selection
   │
   ▼
Model Evaluation
   │
   ▼
Save Trained Model
   │
   ├── predict_model.pkl
   ├── full_model.pkl
   └── label.pkl
   │
   ▼
Disease Prediction
   │
   ▼
Streamlit Web Application
```

---

## How to Run
(all the command is for powershell)

### 1. Clone & Virtual Env & Install

```bash
git clone https://github.com/ayushkumar9576/symptom-driven-disease-prediction.git
cd Disease-Prediction-System

python -m venv Virtual
# Windows:
    Virtual\Scripts\activate
# Linux:
    source Virtual/bin/activate

pip install -r Requirements.txt
```

### 3. Train the model

```bash
python -m Main
```
This will:

- Load the dataset
- Preprocess the data
- Split the dataset
- Train the Random Forest model
- Evaluate the model
- Save the trained model and label encoder

The following files will be generated inside the `model/` directory:

```text
predict_model.pkl
full_model.pkl
label.pkl
evaluation_report.txt
```

---