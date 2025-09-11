import streamlit as st
import joblib
import numpy as np
import pandas as pd
import io
from PIL import Image
import pytesseract
import re
import os
from deep_translator import GoogleTranslator
from gtts import gTTS

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

# --- CUSTOM STYLING FOR PROFESSIONAL UI ---
st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .main-container {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 30px;
        background-color: white;
    }
    .st-emotion-cache-1jmve3k { /* Class for the expander to look cleaner */
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #e0e0e0;
    }
    .summary-card {
        background-color: #e6f7ff; /* Light blue */
        border-left: 5px solid #0072B5;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
    }
    .summary-card h3 {
        color: #0072B5;
        font-size: 2.5em;
        font-weight: bold;
    }
    .summary-card p {
        color: #333333;
        font-size: 1.1em;
    }
    .result-title {
        color: #262626;
    }
    .stButton>button {
        background-color: #0072B5;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #005f9c;
    }
    .welcome-container {
        text-align: center;
        padding-top: 100px;
        padding-bottom: 100px;
    }
    .welcome-container h1 {
        font-size: 3.5em;
        color: #0072B5;
    }
    .welcome-container p {
        font-size: 1.5em;
        color: #555555;
    }
</style>
""", unsafe_allow_html=True)


# --- LANGUAGE CONFIGURATION ---
LANGUAGES = {
    'en': 'English', 'hi': 'Hindi', 'es': 'Spanish', 'fr': 'French',
    'de': 'German', 'ta': 'Tamil', 'te': 'Telugu'
}

def translate_text(text, target_lang='en'):
    if target_lang == 'en' or not text:
        return text
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        translated = translator.translate(text)
        return translated
    except Exception as e:
        st.warning(f"Translation failed for '{text}'. Error: {e}")
        return text

# --- SIDEBAR (NEW LOCATION) ---
with st.sidebar:
    st.title("Settings")
    # Placeholder for a custom logo. Replace 'path/to/your/logo.png' with your actual logo file.
    # st.image("path/to/your/logo.png", width=200)
    st.markdown("## 🌐 Language")
    selected_lang = st.selectbox("Select Language", list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x])


# --- MODEL AND FILE LOADING (REVISED WITH CACHING) ---
@st.cache_resource
def load_model():
    try:
        return joblib.load("models/xgb_model.pkl")
    except FileNotFoundError:
        st.error("❌ Model file missing from the 'models' directory.")
        st.stop()

@st.cache_data
def load_data_files():
    try:
        return joblib.load("models/label_encoder.pkl"), joblib.load("models/important_features.pkl")
    except FileNotFoundError:
        st.error("❌ Required data files are missing from the 'models' directory.")
        st.stop()

# try:
#     pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract-OCR\tesseract.exe"
# except Exception as e:
#     st.warning(f"Tesseract executable not found. OCR functionality may not work. Error: {e}")

model = load_model()
le, feature_names = load_data_files()

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
        
        with st.expander(translate_text("Show Raw OCR Text", selected_lang)):
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
        st.error(f"❌ OCR error: {e}. " + translate_text("Please try again or use manual input.", selected_lang))
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
precautions = {
    "Anemia": [
        "Consult a hematologist for a detailed diagnosis.",
        "Include iron-rich foods like spinach, lentils, and red meat in your diet.",
        "Take prescribed iron and vitamin supplements."
    ],
    "Diabetes": [
        "Consult an endocrinologist or diabetologist.",
        "Monitor your blood glucose levels regularly.",
        "Follow a balanced diet plan and exercise routine.",
        "Reduce your intake of sugary foods and refined carbs."
    ],
    "Leukemia": [
        "This is a serious condition. Immediately consult a hematologist or oncologist.",
        "Follow all medical advice and treatment plans rigorously.",
        "Do not self-medicate.",
        "Get a second opinion from a reputable medical institution if needed."
    ],
    "Lymphoma": [
        "Consult a hematologist or oncologist immediately.",
        "Follow the prescribed treatment plan, which may include chemotherapy or radiation.",
        "Participate in regular follow-up appointments."
    ],
    "Myeloma": [
        "Consult a hematologist or oncologist without delay.",
        "Adhere to your treatment plan.",
        "Stay informed about your condition and treatment options."
    ],
    "Thalasse": [
        "Consult a hematologist for management.",
        "Avoid iron supplements unless specifically prescribed by your doctor.",
        "Get regular blood transfusions if required for severe forms of the condition."
    ],
    "Thromboc": [
        "Consult a hematologist for a proper diagnosis.",
        "Avoid activities that can cause injury or bleeding.",
        "Follow your doctor's recommendations for managing your platelet count."
    ],
    "Healthy": [
        "Maintain a healthy lifestyle with a balanced diet and regular exercise.",
        "Get a routine health check-up once a year.",
        "Avoid smoking and excessive alcohol consumption."
    ]
}
disease_emojis = {
    "Anemia": "🔴", "Diabetes": "🩸", "Healthy": "✅", "Leukemia": "⚪",
    "Lymphoma": "🎗", "Myeloma": "🦴", "Thalasse": "🧬", "Thromboc": "🩹"
}
original_features = [feat for feat in feature_names if feat not in ['WBC_Platelet_Ratio', 'Hemo_x_Hema', 'Glucose_BMI_Interaction']]

# --- TEXT-TO-SPEECH FUNCTION ---
def text_to_speech(text, lang='en'):
    if not text:
        return
    try:
        tts = gTTS(text=text, lang=lang)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        st.audio(audio_bytes, format='audio/mp3', start_time=0, autoplay=True)
    except Exception as e:
        st.error(f"Text-to-speech failed. Error: {e}")

# --- APP STATE MANAGEMENT FOR FRONT PAGE ---
if 'app_state' not in st.session_state:
    st.session_state.app_state = 'welcome'

# --- THE WELCOME PAGE ---
if st.session_state.app_state == 'welcome':
    st.markdown(f"""
    <div class="welcome-container">
        <h1>{translate_text('Blood Disease Predictor', selected_lang)}</h1>
        <p>{translate_text('Analyze your blood report to get insights on various health conditions.', selected_lang)}</p>
        <p>{translate_text('Built with machine learning and medical data.', selected_lang)}</p>
        <br>
        <br>
    </div>
    """, unsafe_allow_html=True)
    if st.button(translate_text("🚀 Start Prediction", selected_lang), use_container_width=True):
        st.session_state.app_state = 'main_app'
        st.rerun()

# --- MAIN APPLICATION LOGIC ---
if st.session_state.app_state == 'main_app':
    with st.container(border=False):
        st.title(translate_text("🩸 Blood Report Analysis", selected_lang))
        st.markdown(f"""
            <p style='font-size: 1.1em;'>{translate_text('Upload your blood report image or enter values manually to predict possible blood disorders.', selected_lang)}</p>
        """, unsafe_allow_html=True)
        
        st.divider()

        # --- USER INPUT FORM WITH TABS (UI ENHANCEMENT) ---
        with st.form("prediction_form"):
            tab1, tab2, tab3 = st.tabs([
                translate_text("📸 Use Camera", selected_lang),
                translate_text("📂 Upload Photo", selected_lang),
                translate_text("✍ Manual Input", selected_lang)
            ])

            input_data = {}
            image_bytes = None

            with tab1:
                st.info(translate_text("Take a photo of your blood report using your device's camera.", selected_lang))
                camera_capture = st.camera_input(translate_text("Capture Image", selected_lang))
                if camera_capture:
                    st.success(translate_text("Photo captured successfully.", selected_lang))
                    image_bytes = camera_capture.getvalue()
            
            with tab2:
                st.info(translate_text("Upload a photo of your blood report from your device.", selected_lang))
                uploaded_file = st.file_uploader(translate_text("Upload image (JPG/PNG)", selected_lang), type=["jpg", "jpeg", "png"])
                if uploaded_file:
                    st.success(translate_text("Image uploaded.", selected_lang))
                    image_bytes = uploaded_file.read()

            with tab3:
                with st.container(border=True):
                    st.markdown(f"{translate_text('Enter your blood test results', selected_lang)}")
                    cols = st.columns(2)
                    for i, feature in enumerate(original_features):
                        with cols[i % 2]:
                            label = friendly_names.get(feature, feature)
                            tip = help_texts.get(feature, "")
                            input_data[feature] = st.number_input(translate_text(label, selected_lang), min_value=0.0, max_value=1000.0, value=0.0, step=0.1, help=translate_text(tip, selected_lang), key=f"manual_{feature}")

            submitted = st.form_submit_button(translate_text("🔍 Predict Disease", selected_lang), type="primary", use_container_width=True)

        # --- PREDICTION LOGIC (ORIGINAL LOGIC WITH IMPROVED UI) ---
        if submitted:
            st.divider()
            with st.spinner(translate_text("Analyzing data and predicting...", selected_lang)):
                final_input_values = {}
                if image_bytes:
                    ocr_values = extract_values_from_image(image_bytes)
                    if not ocr_values:
                        st.warning("⚠ " + translate_text("No valid values found in OCR output. Falling back to manual input.", selected_lang))
                        final_input_values = input_data
                    else:
                        for feat in original_features:
                            final_input_values[feat] = ocr_values.get(feat, input_data.get(feat))
                else:
                    final_input_values = input_data

                if not any(final_input_values.values()):
                    st.error("❌ " + translate_text("No valid values detected. Please enter values or upload a readable image.", selected_lang))
                    st.stop()

                input_df = pd.DataFrame([final_input_values])
                
                with st.expander(translate_text("Show Input Dataframes", selected_lang)):
                    st.write(translate_text("Original input values:", selected_lang))
                    st.dataframe(input_df)
                    
                    input_df['WBC_Platelet_Ratio'] = input_df['White Blood Cells'] / input_df['Platelets']
                    input_df['Hemo_x_Hema'] = input_df['Hemoglobin'] * input_df['Hematocrit']
                    input_df['Glucose_BMI_Interaction'] = input_df['Glucose'] * input_df['BMI']
                    input_df.replace([np.inf, -np.inf], 0, inplace=True)
                    input_df = input_df[feature_names]
                    
                    st.write(translate_text("Final input values for prediction:", selected_lang))
                    st.dataframe(input_df)

                try:
                    probs = model.predict_proba(input_df)
                    prediction = np.argmax(probs, axis=1)
                    predicted_label = le.inverse_transform(prediction)[0]
                    confidence = np.max(probs)
                    
                    st.session_state['prediction'] = predicted_label
                    st.session_state['confidence'] = confidence
                    st.session_state['probs_df'] = pd.DataFrame(probs, columns=le.classes_, index=[translate_text("Probability", selected_lang)])
                    st.session_state['input_df'] = input_df
                except Exception as e:
                    st.error("🚫 " + translate_text(f"Prediction failed: {e}", selected_lang))
                    st.stop()

        # --- DISPLAY RESULTS (IMPROVED UI) ---
        if 'prediction' in st.session_state:
            st.subheader(translate_text("✅ Prediction Results", selected_lang))
            st.divider()
            
            predicted_disease = st.session_state['prediction']
            emoji = disease_emojis.get(predicted_disease, "❓")
            
            st.markdown(f"""
            <div class="summary-card">
                <h3>{translate_text('Predicted Condition:', selected_lang)} {emoji} {translate_text(predicted_disease, selected_lang)}</h3>
                <p>{translate_text(disease_info.get(predicted_disease, 'No additional info available.'), selected_lang)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if predicted_disease == 'Healthy':
                st.balloons()
            else:
                st.error("⚠ " + translate_text("It is highly recommended to consult a medical professional.", selected_lang))
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                full_text_for_speech = f"The predicted condition is {predicted_disease}. {disease_info.get(predicted_disease, 'No additional information.')}"
                if st.button(translate_text("▶ Listen to the results", selected_lang), use_container_width=True):
                    text_to_speech(translate_text(full_text_for_speech, selected_lang), lang=selected_lang)

            

            # --- PRECAUTIONS SECTION ---
            st.markdown("---")
            st.subheader("⚠ " + translate_text("Important Precautions", selected_lang))
            
            if predicted_disease in precautions:
                prec_list = precautions[predicted_disease]
                translated_prec_list = [translate_text(p, selected_lang) for p in prec_list]
                st.markdown(
                    f"""
                    <ul style='font-size: 1.1em;'>
                    {''.join([f'<li>{p}</li>' for p in translated_prec_list])}
                    </ul>
                    """, unsafe_allow_html=True
                )
            else:
                st.info(translate_text("No specific precautions found for this condition. Please consult a doctor for personalized advice.", selected_lang))

            st.markdown("---")
            with st.expander(translate_text("View Full Prediction Probabilities", selected_lang)):
                st.dataframe(st.session_state['probs_df'].style.format("{:.2%}"))