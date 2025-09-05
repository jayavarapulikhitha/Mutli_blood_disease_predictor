import streamlit as st
import joblib
import numpy as np
import pandas as pd
import io
from PIL import Image
import pytesseract
import re
import os

# --- PAGE CONFIGURATION (UI ENHANCEMENT) ---
st.set_page_config(
    page_title="Multi Blood Disease Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS (UI ENHANCEMENT) ---
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css(".streamlit/style.css")

# --- MODEL AND FILE LOADING (ORIGINAL LOGIC) ---
try:
    pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract-OCR\tesseract.exe"
except Exception as e:
    st.warning(f"Tesseract executable not found. OCR functionality may not work. Error: {e}")

try:
    model = joblib.load("models/xgb_model.pkl")
    le = joblib.load("models/label_encoder.pkl")
    feature_names = joblib.load("models/important_features.pkl")
except FileNotFoundError:
    st.error("❌ Required model files are missing from the 'models' directory.")
    st.stop()

# --- HELPER FUNCTIONS (ORIGINAL LOGIC) ---
def preprocess_image(image):
    gray = image.convert("L")
    enhanced = gray.point(lambda x: 0 if x < 140 else 255, '1')
    return enhanced

def extract_values_from_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = preprocess_image(image)
        raw_text = pytesseract.image_to_string(image)
        
        with st.expander("Show Raw OCR Text"):
            st.text(raw_text)

        extracted_values = {}
        feature_mapping = {
            'Platelets': ['platelets', 'plt'], 'C-reactive Protein': ['crp', 'c-reactive protein'],
            'Mean Corpuscular Hemoglobin': ['mch', 'mean corpuscular hemoglobin'], 'White Blood Cells': ['wbc', 'white blood cells'],
            'Hematocrit': ['hct', 'hematocrit'], 'AST': ['ast'], 'BMI': ['bmi'],
            'Glucose': ['glucose'], 'Cholesterol': ['cholesterol'], 'Hemoglobin': ['hemoglobin', 'hgb']
        }

        for feature, keywords in feature_mapping.items():
            for keyword in keywords:
                match = re.search(rf"{keyword}[^0-9]*([\d\.]+)", raw_text, re.IGNORECASE)
                if match:
                    extracted_values[feature] = float(match.group(1))
                    break
        return extracted_values
    except Exception as e:
        st.error(f"❌ OCR error: {e}. Please try again or use manual input.")
        return {}

# --- FEATURE METADATA (ORIGINAL LOGIC) ---
friendly_names = {
    "Platelets": "Platelet Count (×10⁹/L)", "C-reactive Protein": "CRP Level (mg/L)",
    "Mean Corpuscular Hemoglobin": "MCH (pg)", "White Blood Cells": "WBC Count (×10⁹/L)",
    "Hematocrit": "Hematocrit (%)", "AST": "AST (U/L)", "BMI": "Body Mass Index (kg/m²)",
    "Glucose": "Blood Glucose (mg/dL)", "Cholesterol": "Cholesterol (mg/dL)", "Hemoglobin": "Hemoglobin (g/dL)"
}
help_texts = {
    "Platelets": "Normal: 150–450", "C-reactive Protein": "Normal: <10 mg/L",
    "Mean Corpuscular Hemoglobin": "Normal: 27–33 pg", "White Blood Cells": "Normal: 4–11 ×10⁹/L",
    "Hematocrit": "Normal: 36–50%", "AST": "Normal: <40 U/L", "BMI": "Normal: 18.5–24.9",
    "Glucose": "Normal: 70–99 mg/dL", "Cholesterol": "Normal: <200 mg/dL", "Hemoglobin": "Normal: 13–17 (M), 12–15 (F)"
}
disease_info = {
    "Anemia": "Anemia is a condition where you lack enough red blood cells.",
    "Diabetes": "Diabetes affects your blood glucose regulation.",
    "Healthy": "Your report appears healthy. Maintain your lifestyle!",
    "Leukemia": "Leukemia is a cancer of the blood-forming tissues.",
    "Lymphoma": "Lymphoma affects the lymphatic system.",
    "Myeloma": "Myeloma is a cancer of plasma cells.",
    "Thalasse": "Thalassemia causes abnormal hemoglobin.",
    "Thromboc": "Thrombocytopenia means low platelet count."
}
original_features = [feat for feat in feature_names if feat not in ['WBC_Platelet_Ratio', 'Hemo_x_Hema', 'Glucose_BMI_Interaction']]

# --- HEADER AND DESCRIPTION (UI ENHANCEMENT) ---
st.title("🩸 Blood Disease Predictor")
st.markdown("""
    <p style='font-size: 1.1em;'>Upload your blood report image or enter values manually to predict possible blood disorders.</p>
    <hr/>
""", unsafe_allow_html=True)

# --- USER INPUT FORM WITH TABS (UI ENHANCEMENT) ---
with st.form("prediction_form"):
    tab1, tab2, tab3 = st.tabs(["📸 Use Camera", "📂 Upload Photo", "✍️ Manual Input"])

    input_data = {}
    image_bytes = None

    with tab1:
        st.info("Take a photo of your blood report using your device's camera.")
        camera_capture = st.camera_input("Capture Image")
        if camera_capture:
            st.success("Photo captured successfully.")
            image_bytes = camera_capture.getvalue()
    
    with tab2:
        st.info("Upload a photo of your blood report from your device.")
        uploaded_file = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            st.success("Image uploaded.")
            image_bytes = uploaded_file.read()

    with tab3:
        st.info("Enter the values from your blood report manually.")
        cols = st.columns(2)
        for i, feature in enumerate(original_features):
            with cols[i % 2]:
                label = friendly_names.get(feature, feature)
                tip = help_texts.get(feature, "")
                input_data[feature] = st.number_input(label, min_value=0.0, max_value=1000.0, value=0.0, step=0.1, help=tip, key=f"manual_{feature}")

    submitted = st.form_submit_button("🔍 Predict Disease", type="primary", use_container_width=True)

# --- PREDICTION LOGIC (ORIGINAL LOGIC WITH IMPROVED UI) ---
if submitted:
    st.markdown("---")
    
    with st.spinner("Analyzing data and predicting..."):
        final_input_values = {}
        if image_bytes:
            ocr_values = extract_values_from_image(image_bytes)
            if not ocr_values:
                st.warning("⚠ No valid values found in OCR output. Falling back to manual input.")
                final_input_values = input_data
            else:
                for feat in original_features:
                    final_input_values[feat] = ocr_values.get(feat, input_data.get(feat))
        else:
            final_input_values = input_data

        if not any(final_input_values.values()):
            st.error("❌ No valid values detected. Please enter values or upload a readable image.")
            st.stop()

        input_df = pd.DataFrame([final_input_values])
        
        with st.expander("Show Input Dataframes"):
            st.write("Original input values:")
            st.dataframe(input_df)
            
            input_df['WBC_Platelet_Ratio'] = input_df['White Blood Cells'] / input_df['Platelets']
            input_df['Hemo_x_Hema'] = input_df['Hemoglobin'] * input_df['Hematocrit']
            input_df['Glucose_BMI_Interaction'] = input_df['Glucose'] * input_df['BMI']
            input_df.replace([np.inf, -np.inf], 0, inplace=True)
            input_df = input_df[feature_names]
            
            st.write("Final input values for prediction:")
            st.dataframe(input_df)

        try:
            probs = model.predict_proba(input_df)
            prediction = np.argmax(probs, axis=1)
            predicted_label = le.inverse_transform(prediction)[0]
            confidence = np.max(probs)
            
            st.session_state['prediction'] = predicted_label
            st.session_state['confidence'] = confidence
            st.session_state['probs_df'] = pd.DataFrame(probs, columns=le.classes_, index=["Probability"])

        except Exception as e:
            st.error(f"🚫 Prediction failed: {e}")
            st.stop()

# --- DISPLAY RESULTS (UI ENHANCEMENT) ---
if 'prediction' in st.session_state:
    st.subheader("✅ Prediction Results")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"""
            <div class="result-box">
                <p class="result-text">Predicted Condition:</p>
                <h3 class="result-title">🧬 {st.session_state['prediction']}</h3>
            </div>
            """, unsafe_allow_html=True)
        st.info(f"ℹ️ About {st.session_state['prediction']}: {disease_info.get(st.session_state['prediction'], 'No additional info available.')}")

    with col2:
        st.markdown("##### Model Confidence")
        st.metric(label="Probability", value=f"{st.session_state['confidence']:.2%}")
        # --- THE FIX IS ON THE NEXT LINE ---
        st.progress(float(st.session_state['confidence']))
        # ------------------------------------
        
        if st.session_state['prediction'] == 'Healthy':
            st.balloons()
            st.markdown("<p style='text-align: center; color: green;'>🎉 Good news! Stay healthy.</p>", unsafe_allow_html=True)
        else:
            st.error("⚠️ It is highly recommended to consult a medical professional.")
            
    st.markdown("---")
    
    with st.expander("View Full Prediction Probabilities"):
        st.dataframe(st.session_state['probs_df'].style.format("{:.2%}"))