import streamlit as st
import requests

st.set_page_config(page_title="Insurance Fraud Predictor", page_icon="🚗", layout="centered")

st.title("🚨 Auto Insurance Fraud Detection")
st.markdown("Fill in the claim details to check whether it's **Fraudulent** or **Genuine**.")

gender = st.selectbox("Gender", ["male", "female"])
age = st.slider("Age of Insured", 18, 100, 30)
education = st.selectbox("Education Level", ["High School", "College", "Masters", "PhD"])
occupation = st.selectbox("Occupation", ["engineer", "doctor", "teacher", "lawyer", "other"])
hobbies = st.selectbox("Hobbies", ["reading", "sports", "traveling", "gaming"])
accident_severity = st.selectbox("Accident Severity", ["Minor", "Major", "Total Loss"])
accident_type = st.selectbox("Accident Type", ["Rear Collision", "Side Collision", "Head-On", "Other"])
vehicle_cost = st.number_input("Vehicle Cost", min_value=500.0, value=15000.0)
annual_mileage = st.number_input("Annual Mileage", min_value=1000.0, value=10000.0)
total_claim = st.number_input("Total Claim Amount", min_value=0.0, value=5000.0)

input_data = {
    "Gender": gender,
    "Age_Insured": age,
    "Education": education,
    "Occupation": occupation,
    "Hobbies": hobbies,
    "Accident_Severity": accident_severity,
    "Accident_Type": accident_type,
    "Vehicle_Cost": vehicle_cost,
    "Annual_Mileage": annual_mileage,
    "Total_Claim": total_claim,
}

if st.button("🔍 Predict Fraud"):
    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=input_data)
        result = response.json()["prediction"]
        if result == "Fraud":
            st.error(f"🚨 Prediction: {result}")
        else:
            st.success(f"✅ Prediction: {result}")
    except Exception as e:
        st.warning("⚠️ Could not connect to the Flask backend. Make sure it's running.")
        st.text(f"Error: {e}")

