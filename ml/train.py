"""
Accident Severity Prediction - train.py
Real data: 1776093911106_AccidentReports.csv
Target: Severity (Fatal / Grievous Injury / Simple Injury / Damage Only)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
import pickle

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG — update DATA_PATH to your file location
# ─────────────────────────────────────────────
DATA_PATH = r"C:\Users\HP\RiskRadar\ml\data\AccidentReports.csv"

TARGET_COLUMN = "Severity"

# Valid severity classes — everything else is treated as noise and dropped
VALID_SEVERITY = {"Fatal", "Grievous Injury", "Simple Injury", "Damage Only"}

FEATURE_COLUMNS = [
    "Main_Cause",
    "Weather",
    "Road_Type",
    "Collision_Type",
    "Road_Character",
    "Surface_Condition",
    "Road_Condition",
    "Junction_Control",
    "Noofvehicle_involved",
    "Hit_Run",
    "Lane_Type",
    "Year",
    "DISTRICTNAME"
]

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data...")
print("=" * 60)

df = pd.read_csv(DATA_PATH, encoding="latin1")
print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")


# ─────────────────────────────────────────────
# 2. CLEAN TARGET VARIABLE
# ─────────────────────────────────────────────
print("\nSTEP 2: Cleaning target variable...")

before = len(df)
df = df[df[TARGET_COLUMN].isin(VALID_SEVERITY)]
df = df.dropna(subset=[TARGET_COLUMN])
after = len(df)
print(f"  Removed {before - after:,} rows with invalid/noisy Severity values")
print(f"  Remaining rows: {after:,}")
print(f"  Class distribution:\n{df[TARGET_COLUMN].value_counts().to_string()}")


# ─────────────────────────────────────────────
# 3. PREPARE FEATURES
# ─────────────────────────────────────────────
print("\nSTEP 3: Preparing features...")

# Separate numeric and categorical features
CATEGORICAL = [
    "Main_Cause", "Weather", "Road_Type", "Collision_Type",
    "Road_Character", "Surface_Condition", "Road_Condition",
    "Junction_Control", "Hit_Run", "Lane_Type",
    "DISTRICTNAME"
]
NUMERIC = ["Noofvehicle_involved", "Year"]

df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

# Fill missing/noise values with "Unknown" instead of dropping
noise = ["Not Applicable", "not applicable", "N/A", "NA", "None", ""]
for col in CATEGORICAL:
    df[col] = df[col].replace(noise, "Unknown")
    df[col] = df[col].fillna("Unknown")

# Only drop rows where numeric columns are missing
for col in NUMERIC:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=NUMERIC)

print(f"  Rows after cleaning: {len(df):,}")

# Encode categoricals
label_encoders = {}
for col in CATEGORICAL:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"  Encoded '{col}' → {len(le.classes_)} categories")

# Encode target with meaningful order
severity_order = ["Damage Only", "Simple Injury", "Grievous Injury", "Fatal"]
target_encoder = OrdinalEncoder(categories=[severity_order])
y = target_encoder.fit_transform(df[[TARGET_COLUMN]]).ravel().astype(int)
X = df[FEATURE_COLUMNS].values

print(f"\n  Feature matrix shape: {X.shape}")
print(f"  Target classes: {severity_order}")

# Save encoders
with open(os.path.join(OUTPUT_DIR, "label_encoders.pkl"), "wb") as f:
    pickle.dump({"features": label_encoders, "target": target_encoder}, f)
print(f"  Saved encoders → {OUTPUT_DIR}/label_encoders.pkl")


# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
print("\nSTEP 4: Splitting data (70/30)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")


# ─────────────────────────────────────────────
# 5. TRAIN MODELS
# ─────────────────────────────────────────────
print("\nSTEP 5: Training models...")

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=15, class_weight="balanced",
        random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=500, class_weight="balanced", random_state=42
    ),
}

# Try XGBoost and LightGBM if installed
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1
    )
    print("  XGBoost detected — added to model list")
except ImportError:
    print("  XGBoost not installed — skipping (pip install xgboost)")

try:
    from lightgbm import LGBMClassifier
    models["LightGBM"] = LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    print("  LightGBM detected — added to model list")
except ImportError:
    print("  LightGBM not installed — skipping (pip install lightgbm)")


results = {}
best_model = None
best_f1 = 0

for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(
        y_test, y_pred,
        target_names=severity_order,
        output_dict=True
    )

    results[name] = {
        "f1_weighted": round(f1, 4),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall": round(report[cls]["recall"], 4),
                "f1": round(report[cls]["f1-score"], 4),
            }
            for cls in severity_order
        }
    }

    print(f"    F1 (weighted): {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=severity_order))

    if f1 > best_f1:
        best_f1 = f1
        best_model = (name, model)


# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE (best model)
# ─────────────────────────────────────────────
print("\nSTEP 6: Feature importance (best model)...")
best_name, best_clf = best_model
print(f"  Best model: {best_name} (F1={best_f1:.4f})")

if hasattr(best_clf, "feature_importances_"):
    importances = best_clf.feature_importances_
    feat_imp = sorted(
        zip(FEATURE_COLUMNS, importances),
        key=lambda x: x[1], reverse=True
    )
    print("\n  Feature Importances:")
    for feat, imp in feat_imp:
        bar = "█" * int(imp * 40)
        print(f"    {feat:<30} {imp:.4f}  {bar}")
    results["feature_importance"] = {f: round(float(i), 4) for f, i in feat_imp}


# ─────────────────────────────────────────────
# 7. HOTSPOT ANALYSIS
# ─────────────────────────────────────────────
print("\nSTEP 7: Generating hotspot report...")

df_full = pd.read_csv(DATA_PATH, encoding="latin1")
df_full = df_full[df_full[TARGET_COLUMN].isin(VALID_SEVERITY)]

hotspots = (
    df_full.groupby("DISTRICTNAME")
    .agg(
        total_accidents=("Crime_No", "count"),
        fatal_count=(TARGET_COLUMN, lambda x: (x == "Fatal").sum()),
    )
    .assign(fatal_rate=lambda d: (d["fatal_count"] / d["total_accidents"] * 100).round(2))
    .sort_values("fatal_rate", ascending=False)
    .head(15)
    .reset_index()
)

hotspot_path = os.path.join(OUTPUT_DIR, "hotspots.json")
hotspots.to_json(hotspot_path, orient="records", indent=2)
print(f"  Top 5 high-risk districts by fatal rate:")
print(hotspots[["DISTRICTNAME", "total_accidents", "fatal_count", "fatal_rate"]].head(5).to_string(index=False))
print(f"  Saved → {hotspot_path}")


# ─────────────────────────────────────────────
# 8. SAVE RESULTS
# ─────────────────────────────────────────────
print("\nSTEP 8: Saving results...")

metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  Saved metrics → {metrics_path}")

model_path = os.path.join(OUTPUT_DIR, "best_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(best_clf, f)
print(f"  Saved best model ({best_name}) → {model_path}")

print("\n" + "=" * 60)
print(f"  DONE — Best model: {best_name} | F1: {best_f1:.4f}")
print("=" * 60)