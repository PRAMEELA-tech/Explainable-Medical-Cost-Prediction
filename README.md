# 💊 Explainable Medical Insurance Cost Prediction System

## 📌 Project Overview

This project is an end-to-end Machine Learning system that predicts medical insurance costs based on demographic and lifestyle factors.

The system follows an **Explainability-First approach**, meaning every prediction is accompanied by a clear, human-readable explanation derived from SHAP (SHapley Additive Explanations).

The project includes:

- Data preprocessing  
- Feature engineering  
- Model training & comparison  
- Hyperparameter tuning  
- Global & Local explainability  
- Structured explanation extraction  
- Natural language explanation engine  
- FastAPI backend deployment  
- Streamlit frontend UI  

---

## 🎯 Problem Statement

Predict medical insurance charges using:

- Age  
- Sex  
- BMI  
- Number of children  
- Smoking status  
- Region  

### Output:
- Predicted medical insurance cost  
- Explanation of key contributing factors  

---

## 🧠 Key Features

✔ Regression modeling (Linear, Ridge, Random Forest, Gradient Boosting)  
✔ Light hyperparameter tuning  
✔ SHAP-based explainability  
✔ Structured explanation extraction  
✔ Rule-based natural language explanation  
✔ FastAPI backend  
✔ Streamlit UI  
✔ Clean separation between training and deployment  

---

## 🏗 Project Architecture

User → Streamlit UI → FastAPI Backend → ML Model → SHAP → Text Explanation → UI Display


### 🔹 Architecture Layers

### 1️⃣ Training Layer
- Exploratory Data Analysis (EDA)  
- Data cleaning & encoding  
- Feature engineering  
- Baseline & advanced regression models  
- Hyperparameter tuning  
- Model evaluation  

### 2️⃣ Explainability Layer
- SHAP TreeExplainer  
- Global feature importance analysis  
- Local prediction explanations  
- Structured explanation extraction  
- Natural language explanation generation  

### 3️⃣ Deployment Layer
- FastAPI backend for prediction & explanation APIs  
- Streamlit frontend for user interaction  
- Separate environments for backend and frontend  

---

## 📁 Project Structure

Medical_Cost_Project/
│
├── api/ # FastAPI backend
│ ├── app.py
│ ├── requirements.txt
│ └── models/
│ ├── final_model.pkl
│ ├── preprocessor.pkl
│ └── feature_names.pkl
│
├── ui/ # Streamlit frontend
│ ├── app.py
│ └── requirements.txt
│
├── data/ # Dataset (not included in repo)
│
├── models/ # Training artifacts & evaluation proof
│
├── notebooks/ # Phase-wise ML development notebooks
│
└── README.md


---

## 📊 Dataset

The dataset is **NOT included** in this repository to keep it lightweight.

Download it from Kaggle:

👉 https://www.kaggle.com/datasets/mirichoi0218/insurance

After downloading, place the dataset inside:

Medical_Cost_Project/data/


---

## 📁 Training Artifacts

The `models/` directory contains:

- Optimized regression models  
- Evaluation metrics  
- Model comparison results  
- SHAP explanation outputs  
- Structured explanation artifacts  

These files are included for academic transparency and reproducibility.

---

## 🚀 How to Run the Project

This project consists of two independent services:

- Backend (FastAPI)
- Frontend (Streamlit)

They must be run in separate terminals.

---

### 1️⃣ Run FastAPI Backend

```bash
cd Medical_Cost_Project/api
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```
API documentation available at:
http://localhost:8000/docs

Open a new terminal:
```bash
cd Medical_Cost_Project/ui
pip install -r requirements.txt
streamlit run app.py
```
---

# 🏆 Academic Highlights
- Full machine learning lifecycle implementation
- Model comparison & hyperparameter tuning
- Explainability as a core design principle
- Structured explanation pipeline
- Industry-style backend–frontend separation
- Deployment-ready architecture
---


# ⚠ Limitations
- Small dataset (~1338 records)
- Limited health-related features
- No real-time insurance database integration
---

# 🔮 Future Improvements
- Add SHAP visual plots inside UI
- Deploy on cloud platform (AWS / Render / Railway)
- Add Docker containerization
- Extend to larger healthcare datasets
----
# 📌 Repository Notes
- Dataset excluded to keep repository lightweight.
- Kaggle API credentials are not included for security reasons.
- Backend and frontend use separate requirements.txt files for modular architecture.
---

## Author

Ginni Prameela 
B.Tech CSE  
Explainable AI Project
