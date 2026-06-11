"""Train RandomForest with balanced weights on raw data."""
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import os

# Load raw data
data_path = "data/raw/student dropout.csv"
df = pd.read_csv(data_path)
print(f"Loaded data: {df.shape}")

# Target column mapping
df['Target'] = df['Dropped_Out'].map({False: 0, True: 1})
print(f"Class distribution: {df['Target'].value_counts().to_dict()}")

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    if col != 'Target':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Prepare features
X = df.drop('Target', axis=1)
y = df['Target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Train RandomForest with balanced class weights
print("\nTraining RandomForest with class_weight='balanced'...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Save model
output_path = "mlruns/artifacts/1/models/random_forest_balanced/model.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "wb") as f:
    pickle.dump(model, f)
print(f"\nModel saved to: {output_path}")

# Save label encoders
encoders_path = "mlruns/artifacts/1/models/random_forest_balanced/encoders.pkl"
with open(encoders_path, "wb") as f:
    pickle.dump(label_encoders, f)
print(f"Encoders saved to: {encoders_path}")

# Show feature names
print(f"\nFeature names: {list(X.columns)}")
