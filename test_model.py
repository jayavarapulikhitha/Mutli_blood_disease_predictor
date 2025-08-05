import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load and clean data
df = pd.read_csv('data/augmented_blood_data.csv')
df.dropna(inplace=True)

# Encode target
le = LabelEncoder()
df['Disease'] = le.fit_transform(df['Disease'])

# Features & target
X = df.drop('Disease', axis=1)
y = df['Disease']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# XGBoost Classifier
xgb = XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y)), eval_metric='mlogloss')

# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3],
    'min_child_weight': [1, 3, 5]
}

# Randomized Search
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=50,
    cv=3,
    scoring='accuracy',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit model
random_search.fit(X_train, y_train)

# Best model
best_model = random_search.best_estimator_

# Save model and encoder
joblib.dump(best_model, 'models/xgb_model.pkl')
joblib.dump(le, 'models/label_encoder.pkl')

# Evaluate
y_pred = best_model.predict(X_val)
print("✅ Tuned Accuracy:", accuracy_score(y_val, y_pred))
print("✅ Classification Report:\n", classification_report(y_val, y_pred, target_names=le.classes_))
