import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# Load data
df = pd.read_csv('data/augmented_blood_data.csv')
df.dropna(inplace=True)

# Balance dataset
target_count = 300
balanced = []
for label in df['Disease'].unique():
    group = df[df['Disease'] == label]
    if len(group) < target_count:
        upsampled = resample(group, replace=True, n_samples=target_count, random_state=42)
        balanced.append(upsampled)
    else:
        balanced.append(group.sample(n=target_count, random_state=42))
df = pd.concat(balanced).sample(frac=1, random_state=42).reset_index(drop=True)

# Encode target
le = LabelEncoder()
df['Disease'] = le.fit_transform(df['Disease'])

# Split
X = df.drop(columns=['Disease'])
y = df['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Feature importance (use top 10)
initial_model = XGBClassifier(eval_metric='mlogloss')
initial_model.fit(X_train, y_train)
importances = pd.Series(initial_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10).index.tolist()

# Use only top 10 features
X_train = X_train[top_features]
X_test = X_test[top_features]

# Model
model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(le.classes_),
    eval_metric='mlogloss'
)

# Train with basic tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
search = RandomizedSearchCV(model, param_grid, n_iter=5, scoring='accuracy', cv=cv, random_state=42)
search.fit(X_train, y_train)

best_model = search.best_estimator_

# Test
y_pred = best_model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Debug Preview
print("\n🔍 Sample Predictions:")
for i in range(5):
    actual = le.inverse_transform([y_test.iloc[i]])[0]
    predicted = le.inverse_transform([y_pred[i]])[0]
    print(f"Row {i+1} → True: {actual} | Predicted: {predicted}")

# Save
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/xgb_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")
joblib.dump(top_features, "models/important_features.pkl")
print("\n✅ Model and features saved.")
