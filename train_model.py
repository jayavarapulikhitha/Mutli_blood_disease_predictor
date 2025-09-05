import pandas as pd
import numpy as np # Import numpy to handle infinity
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# --- Step 1: Load Data and Perform Identical Feature Engineering ---
print("✅ Loading data and applying feature engineering...")
df = pd.read_csv('data/balanced_blood_data.csv')
df.dropna(inplace=True)

# --- Feature Engineering (Must be IDENTICAL to the training script) ---
df['WBC_Platelet_Ratio'] = df['White Blood Cells'] / df['Platelets']
df['Hemo_x_Hema'] = df['Hemoglobin'] * df['Hematocrit']
df['Glucose_BMI_Interaction'] = df['Glucose'] * df['BMI']

# --- Handle Potential Infinite Values (Crucial for consistency) ---
df.replace([np.inf, -np.inf], 0, inplace=True)
print("✅ Feature engineering and data cleaning complete.")


# --- Step 2: Load Model and Supporting Files ---
print("✅ Loading model and required files...")
try:
    le = joblib.load('models/label_encoder.pkl')
    model = joblib.load('models/xgb_model.pkl')
    feature_names = joblib.load('models/important_features.pkl')
except FileNotFoundError:
    print("❌ Error: Model files not found in the 'models' directory. Please run the training script first.")
    exit()


# --- Step 3: Prepare Data for Prediction ---
# Ensure all required features are present before selection
X = df[feature_names]
y = le.transform(df['Disease'])


# --- Step 4: Split Data and Evaluate ---
# We split again with the same random_state to get the identical test set
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("✅ Making predictions on the test set...")
# Predict and evaluate
y_pred = model.predict(X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))