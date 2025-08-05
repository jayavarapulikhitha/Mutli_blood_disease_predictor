from sklearn.ensemble import IsolationForest
import pandas as pd
# Load data
df = pd.read_csv('data/augmented_blood_data.csv')
df.dropna(inplace=True)

# Separate features and target
X = df.drop(columns=['Disease'])
y = df['Disease']

# Detect outliers
iso = IsolationForest(contamination=0.02, random_state=42)
outliers = iso.fit_predict(X)

# Remove outliers (-1 means outlier)
df_filtered = df[outliers != -1]

# Save filtered dataset
df_filtered.to_csv('data/cleaned_blood_data.csv', index=False)
print("✅ Outliers removed and cleaned data saved to 'data/cleaned_blood_data.csv'")
