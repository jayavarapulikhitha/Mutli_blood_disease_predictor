# config.py

LANGUAGES = {
    'en': 'English', 'hi': 'Hindi', 'es': 'Spanish', 'fr': 'French',
    'de': 'German', 'ta': 'Tamil', 'te': 'Telugu'
}

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
    "Lymphoma": "🎗️", "Myeloma": "🦴", "Thalasse": "🧬", "Thromboc": "🩹"
}