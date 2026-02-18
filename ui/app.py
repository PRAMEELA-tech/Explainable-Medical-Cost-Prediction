import streamlit as st
import requests

st.set_page_config(
    page_title="Medical Cost Prediction",
    layout="centered"
)

st.title("💊 Medical Insurance Cost Prediction")
st.write("This UI connects to a FastAPI backend for prediction and explanation.")

# -----------------------------
# Input Form
# -----------------------------
with st.form("medical_form"):
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox(
        "Region",
        ["northeast", "northwest", "southeast", "southwest"]
    )

    submitted = st.form_submit_button("Predict Medical Cost")

# -----------------------------
# On Submit
# -----------------------------
if submitted:
    payload = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }

    try:
        pred_res = requests.post("http://localhost:8000/predict", json=payload)
        exp_res = requests.post("http://localhost:8000/explain", json=payload)

        if pred_res.status_code == 200 and exp_res.status_code == 200:
            pred_data = pred_res.json()
            exp_data = exp_res.json()

            st.success(f"💰 Predicted Medical Cost: ₹ {pred_data['predicted_cost']}")

            st.subheader("🧠 Explanation")
            for k, v in exp_data["explanation"].items():
                st.write(f"**{k.capitalize()}**: {v}")

            st.info(exp_data["summary"])

        else:
            st.error("Backend API error. Check FastAPI server.")

    except Exception as e:
        st.error(f"Connection error: {e}")
