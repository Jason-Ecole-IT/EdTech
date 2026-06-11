"""Train a RandomForest with balanced class weights to fix calibration."""
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import os

# Load data
data_path = "data/processed/model_comparison.csv"
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found")
    exit(1)

df = pd.read_csv(data_path)
print(f"Loaded data: {df.shape}")

# Prepare features - drop non-feature columns
feature_cols = [col for col in df.columns if col not in ['Target', 'student_id']]
X = df[feature_cols]
y = df["Target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Class distribution - Train: {y_train.value_counts().to_dict()}, Test: {y_test.value_counts().to_dict()}")

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

# Show sample predictions on test data
print("\n=== Sample Predictions ===")
sample_df = X_test.head(10).copy()
sample_df["actual"] = y_test.head(10).values
sample_df["predicted"] = y_pred[:10]
sample_df["probability"] = y_proba[:10]
print(sample_df[["actual", "predicted", "probability"]].to_string())
