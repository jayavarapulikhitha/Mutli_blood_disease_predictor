import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import os

# --- Step 1: Load Data and Perform Feature Engineering ---
print("✅ Loading data and applying feature engineering...")
df = pd.read_csv('data/balanced_blood_data.csv')
df.dropna(inplace=True)

# --- Original Feature Engineering ---
df['WBC_Platelet_Ratio'] = df['White Blood Cells'] / df['Platelets']
df['Hemo_x_Hema'] = df['Hemoglobin'] * df['Hematocrit']
df['Glucose_BMI_Interaction'] = df['Glucose'] * df['BMI']

# --- NEW: Targeted Anemia Feature Engineering ---
# This assumes your data is scaled between 0 and 1.
# By subtracting them from 1, we create a score where higher values mean higher
# probability of anemia.
df['Anemia_Score'] = (1 - df['Hemoglobin']) + (1 - df['Hematocrit']) + (1 - df['Red Blood Cells'])

# --- Handle Potential Infinite Values ---
df.replace([np.inf, -np.inf], 0, inplace=True)
print("✅ Feature engineering and data cleaning complete.")


# --- Step 2: Prepare Data for Training ---
# Separate features (X) and target (y)
X = df.drop('Disease', axis=1)
y = df['Disease']

# Get the list of all feature names for saving
feature_names = X.columns.tolist()

# Encode the disease labels to numerical values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# --- Step 3: Train the XGBoost Model ---
print("✅ Training the XGBoost model...")
xgb_model = XGBClassifier(objective='multi:softprob', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
print("✅ Model training complete.")


# --- Step 4: Save the Trained Model and Supporting Files ---
print("✅ Saving model and supporting files...")
# Create a 'models' directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the model with the new feature in the filename
joblib.dump(xgb_model, 'models/xgb_model_with_anemia_score.pkl')
# Save the label encoder
joblib.dump(le, 'models/label_encoder.pkl')
# Save the list of feature names, including the new one
joblib.dump(feature_names, 'models/important_features_with_anemia_score.pkl')

print("✅ All files saved successfully. You can now run the testing script.")