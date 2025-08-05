import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model, encoder, and top features
model = joblib.load("models/xgb_model.pkl")
le = joblib.load("models/label_encoder.pkl")
feature_names = joblib.load("models/important_features.pkl")

st.set_page_config(page_title="Multi Blood Disease Predictor", layout="centered")

st.markdown("""
    <h2 style='text-align: center; color: #2c3e50;'>🔬 Multi-Blood Disease Predictor</h2>
    <p style='text-align: center;'>Enter your blood test values from your report to predict the disease.</p>
    <hr style='border: 1px solid #ccc;'/>
""", unsafe_allow_html=True)

# 🔹 Mapping for user-friendly display names
friendly_names = {
    "Platelets": "Platelet Count (×10⁹/L)",
    "C-reactive Protein": "CRP Level (mg/L)",
    "Mean Corpuscular Hemoglobin": "MCH (pg)",
    "White Blood Cells": "WBC Count (×10⁹/L)",
    "Hematocrit": "Hematocrit (%)",
    "AST": "AST (U/L)",
    "BMI": "Body Mass Index (kg/m²)",
    "Glucose": "Blood Glucose (mg/dL)",
    "Cholesterol": "Cholesterol (mg/dL)",
    "Hemoglobin": "Hemoglobin (g/dL)"
}

# 🔹 Help text for inputs
help_texts = {
    "Platelets": "Check your CBC report. Normal: 150–450",
    "C-reactive Protein": "Inflammation marker. Normal: <10 mg/L",
    "Mean Corpuscular Hemoglobin": "Avg hemoglobin per red cell. Normal: 27–33 pg",
    "White Blood Cells": "WBC count. Normal: 4–11 ×10⁹/L",
    "Hematocrit": "RBC volume %. Normal: 36–50%",
    "AST": "Liver enzyme. Normal: <40 U/L",
    "BMI": "Body Mass Index = weight/height². Normal: 18.5–24.9",
    "Glucose": "Fasting blood sugar. Normal: 70–99 mg/dL",
    "Cholesterol": "Total cholesterol. Normal: <200 mg/dL",
    "Hemoglobin": "Hemoglobin level. Normal: 13–17 (male), 12–15 (female)"
}

# 🔹 Disease Information to Display After Prediction
disease_info = {
    "Anemia": "Anemia occurs when your body lacks enough healthy red blood cells to carry oxygen. Common symptoms include fatigue, weakness, and pale skin.",
    "Diabetes": "Diabetes affects how your body processes blood glucose. It's important to manage through diet, exercise, and sometimes medication.",
    "Healthy": "Your test results suggest a healthy profile. Continue maintaining a balanced diet, regular exercise, and routine checkups.",
    "Leukemia": "Leukemia is a type of cancer of the blood or bone marrow. Early detection and treatment are crucial.",
    "Lymphoma": "Lymphoma affects the lymphatic system. Symptoms may include swollen lymph nodes, fatigue, and weight loss.",
    "Myeloma": "Multiple Myeloma is a cancer of plasma cells in the bone marrow. It can cause bone pain and fatigue.",
    "Thalasse": "Thalassemia is a genetic blood disorder causing the body to produce abnormal hemoglobin. It can lead to anemia.",
    "Thromboc": "Thrombocytopenia means low platelet count, which can cause excessive bleeding or bruising. Causes range from infections to medications."
}

# Input form
with st.form("prediction_form"):
    st.subheader("Enter Your Blood Test Results")

    input_dict = {}

    for feature in feature_names:
        label = friendly_names.get(feature, feature)
        help_text = help_texts.get(feature, "")

        if feature.lower() == 'bmi':
            val = st.number_input(f"{label}", min_value=10.0, max_value=60.0, value=22.0, step=0.1, help=help_text)
        elif feature.lower() == 'glucose':
            val = st.number_input(f"{label}", min_value=50.0, max_value=300.0, value=90.0, step=1.0, help=help_text)
        else:
            val = st.number_input(f"{label}", min_value=0.0, max_value=1000.0, value=50.0, step=1.0, help=help_text)

        input_dict[feature] = val

    submitted = st.form_submit_button("🔍 Predict Disease")

# Prediction
if submitted:
    input_df = pd.DataFrame([input_dict])[feature_names]

    st.write("🔎 Input DataFrame sent to model:", input_df)

    prediction = model.predict(input_df)
    st.write("Raw prediction (encoded):", prediction)

    predicted_label = le.inverse_transform(prediction)[0]
    st.write("Decoded label classes:", list(le.classes_))

    st.success(f"🧬 **Predicted Disease:** {predicted_label}")

    info = disease_info.get(predicted_label, "No additional information available.")
    st.info(f"ℹ️ **About {predicted_label}:**\n\n{info}")
