import pandas as pd
import numpy as np # Import numpy to handle infinity
import os
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

# --- Step 1: Load, Clean, and Engineer Data ---
print("✅ Loading and cleaning the data...")
df = pd.read_csv('data/balanced_blood_data.csv')
df.dropna(inplace=True)

# --- Feature Engineering ---
print("✅ Engineering new features...")
df['WBC_Platelet_Ratio'] = df['White Blood Cells'] / df['Platelets']
df['Hemo_x_Hema'] = df['Hemoglobin'] * df['Hematocrit']
df['Glucose_BMI_Interaction'] = df['Glucose'] * df['BMI']

# --- Handle Potential Infinite Values ---
# This is crucial to prevent the XGBoost error
print("✅ Replacing infinite values after division...")
df.replace([np.inf, -np.inf], 0, inplace=True)

# Encode the target
le = LabelEncoder()
df['Disease'] = le.fit_transform(df['Disease'])


# --- Step 2: Split Data ---
# This part now includes the new engineered features automatically
X = df.drop(columns=['Disease'])
y = df['Disease']
feature_names = X.columns.tolist() # This will now save the new feature names as well

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


# --- Step 3: Hyperparameter Tuning with Reduced Search Space ---
print("✅ Training and tuning the model...")
model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(le.classes_),
    eval_metric='mlogloss',
)

param_grid = {
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1],
    'min_child_weight': [1, 3]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=10,
    cv=cv,
    scoring='f1_macro',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_


# --- Step 4: Evaluate Model ---
print("\n✅ Evaluating the final model...")
y_pred = best_model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ F1-Macro Score:", f1_score(y_test, y_pred, average='macro'))
print("✅ Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))


# --- Step 5: Save Model Files ---
print("\n✅ Saving model and files...")
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/xgb_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")
joblib.dump(feature_names, "models/important_features.pkl")

print("\n✅ Model training complete. Files saved in 'models' folder.")