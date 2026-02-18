from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import shap

app = FastAPI(title="Medical Cost Prediction API")

# ---------------------------
# Root endpoint
# ---------------------------
@app.get("/")
def root():
    return {
        "status": "Medical Cost Prediction API is running",
        "endpoints": ["/predict", "/explain", "/docs"]
    }

# ---------------------------
# Load artifacts
# ---------------------------
model = joblib.load("models/final_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")
feature_names = joblib.load("models/feature_names.pkl")

explainer = shap.TreeExplainer(model)

# ---------------------------
# Input schema
# ---------------------------
class PatientInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

# ---------------------------
# Helper
# ---------------------------
def preprocess_input(data: PatientInput):
    df = pd.DataFrame([data.dict()])
    return preprocessor.transform(df)

# ---------------------------
# /predict
# ---------------------------
@app.post("/predict")
def predict_cost(data: PatientInput):
    X = preprocess_input(data)
    pred = model.predict(X)[0]

    return {
        "predicted_cost": round(float(pred), 2),
        "confidence_context": "Prediction based on trained regression model"
    }

# ---------------------------
# /explain (CLEAN SHAP)
# ---------------------------
@app.post("/explain")
def explain_prediction(data: PatientInput):
    X = preprocess_input(data)
    pred = model.predict(X)[0]

    shap_values = explainer.shap_values(X)[0]

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values
    })

    explanation = {}

    # AGE
    age_shap = shap_df[shap_df["feature"] == "age"]["shap_value"].values[0]
    explanation["age"] = (
        "Lower age reduces the medical cost"
        if age_shap < 0 else
        "Higher age increases the medical cost"
    )

    # BMI
    bmi_shap = shap_df[shap_df["feature"] == "bmi"]["shap_value"].values[0]
    explanation["bmi"] = (
        "BMI is within a safer range"
        if bmi_shap < 0 else
        "Higher BMI increases the medical cost"
    )

    # SMOKER
    smoker_yes = shap_df[shap_df["feature"] == "smoker_yes"]["shap_value"].values[0]
    explanation["smoker"] = (
        "Non-smoker status significantly reduces the cost"
        if smoker_yes < 0 else
        "Smoking status significantly increases the cost"
    )

    # REGION (pick dominant region SHAP)
    region_df = shap_df[shap_df["feature"].str.startswith("region_")]
    top_region = region_df.sort_values(
        by="shap_value", key=np.abs, ascending=False
    ).iloc[0]

    explanation["region"] = (
        "Region contributes to lower average cost"
        if top_region.shap_value < 0 else
        "Region contributes to higher average cost"
    )

    summary = (
        "Overall, the predicted medical cost is lower than average due to healthy lifestyle indicators."
        if sum(shap_values) < 0 else
        "Overall, the predicted medical cost is higher than average due to risk factors."
    )

    return {
        "predicted_cost": round(float(pred), 2),
        "explanation": explanation,
        "summary": summary
    }
