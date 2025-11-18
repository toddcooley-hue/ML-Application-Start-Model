import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os

# ---------- Paths ----------
RAW_PATH = "data/raw/funnel_export.csv"
MODEL_PATH = "models/apply_model.pkl"
OUTPUT_PATH = "output/2026_predictions_python.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("output", exist_ok=True)

# ---------- 1. Load & clean column names ----------
df = pd.read_csv(RAW_PATH)

# Clean column names: similar to janitor::clean_names()
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(r"[^\w]+", "_", regex=True)
      .str.strip("_")
)

print(df.columns.tolist())  # TEMP: sanity check once


# ---------- 2. Feature engineering ----------


# ---------- 2. Feature engineering ----------

# Parse ISO timestamps like 2022-10-14T15:14:17+00:00
df["date_of_inquiry"] = pd.to_datetime(
    df["date_of_inquiry"],
    errors="coerce",
    utc=True
)
df["date_of_inquiry"] = df["date_of_inquiry"].dt.tz_convert(None)

# Days since inquiry
today = pd.Timestamp.today().normalize()
df["inquiry_age"] = (today - df["date_of_inquiry"]).dt.days

# Extract entry_year from the entry term field
df["entry_year"] = (
    df["active_term_calculated"]
        .astype(str)
        .str.extract(r"(\d{4})", expand=False)
)
df["entry_year"] = pd.to_numeric(df["entry_year"], errors="coerce")

# Target variable (0/1)
df["applied"] = pd.to_numeric(df["target_variable"], errors="coerce")

# Confirm existence of renamed fields
# major, recruitment_source must exist after column cleaning

df = df.dropna(subset=["entry_year", "applied"])






# ---------- 3. Split into train (<=2025) and predict (==2026) ----------

train_df = df[df["entry_year"] <= 2025].copy()
predict_df = df[df["entry_year"] == 2026].copy()

# If there are no 2026 records yet, bail gracefully
if predict_df.empty:
    print("No 2026 records found yet. Check active_term_calculated / entry_year.")
    raise SystemExit

# ---------- 3b. Handle missing values in features ----------

feature_cols = [
    "entry_year",
    "major",
    "active_term_calculated",
    "recruitment_source",
    "engagement_score",
    "inquiry_age",
]

cat_cols = ["major", "active_term_calculated", "recruitment_source", "engagement_score"]
num_cols = ["entry_year", "inquiry_age"]

# Fill numeric NaNs with median of TRAINING data
for col in num_cols:
    median_val = train_df[col].median()
    train_df[col] = train_df[col].fillna(median_val)
    predict_df[col] = predict_df[col].fillna(median_val)

# Fill categorical NaNs with "Unknown"
for col in cat_cols:
    train_df[col] = train_df[col].fillna("Unknown")
    predict_df[col] = predict_df[col].fillna("Unknown")


# ---------- 4. Build features & model (simple, solid) ----------

feature_cols = [
    "entry_year",
    "major",
    "active_term_calculated",
    "recruitment_source",
    "engagement_score",
    "inquiry_age",
]



# Categorical vs numeric
cat_cols = ["major", "active_term_calculated", "recruitment_source", "engagement_score"]
num_cols = ["entry_year", "inquiry_age"]


# One-hot encode categoricals
X = pd.get_dummies(train_df[feature_cols], columns=cat_cols, drop_first=True)
y = train_df["applied"]

# Train/validation split to sanity-check performance
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Gradient Boosting is a nice middle ground: strong and not too finicky
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Validation AUC
val_probs = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_probs)
print(f"Validation AUC: {auc:.3f}")

# Refit on full training data
model.fit(X, y)

# Save model and feature columns
joblib.dump({"model": model, "feature_columns": X.columns.tolist()}, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# ---------- 5. Score 2026 records ----------

X_2026 = pd.get_dummies(predict_df[feature_cols], columns=cat_cols, drop_first=True)

# Align columns with training matrix (add any missing columns as 0)
X_2026 = X_2026.reindex(columns=X.columns, fill_value=0)

probs_2026 = model.predict_proba(X_2026)[:, 1]

# Build output table
out_cols = [
    "element_id",
    "first_name",
    "last_name",
    "entry_year",
    "major",
    "active_term_calculated",
    "recruitment_source",
    "engagement_score",
    "inquiry_age",
]
output = predict_df[out_cols].copy()
output["prob_apply"] = probs_2026

output = output.sort_values("prob_apply", ascending=False)

output.to_csv(OUTPUT_PATH, index=False)
print(f"2026 predictions written to {OUTPUT_PATH}")
