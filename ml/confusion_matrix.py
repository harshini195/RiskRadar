"""
confusion_matrix.py
--------------------
Run this AFTER train.py has finished (it needs outputs/best_model.pkl
and outputs/feature_columns.pkl to already exist).

It does NOT modify train.py. It reproduces the exact same data loading,
cleaning, feature-engineering and train/test split (same random_state=42,
same stratify=y) so it lands on the identical X_test / y_test that
train.py evaluated on, then loads the pickled best model and prints /
saves a confusion matrix.

Usage:
    python train.py               # produces outputs/best_model.pkl etc.
    python confusion_matrix.py    # produces outputs/confusion_matrix.png + .json
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── same config as train.py ──────────────────────────────────
DATA_PATH     = "data/cleaned_accidents_full.csv"
TARGET_COLUMN = "risk_level"
OUTPUT_DIR    = "outputs"
RISK_LABELS   = {0: "Low", 1: "Medium", 2: "High"}

print("Loading saved model + feature list...")
with open(os.path.join(OUTPUT_DIR, "best_model.pkl"), "rb") as f:
    best_clf = pickle.load(f)
with open(os.path.join(OUTPUT_DIR, "feature_columns.pkl"), "rb") as f:
    available_features = pickle.load(f)

# ── reproduce train.py STEP 1-3: load + filter + impute ─────
df = pd.read_csv(DATA_PATH, encoding="latin1")
df = df[df[TARGET_COLUMN].isin([0, 1, 2])].copy()

missing_geo_mask = df["Latitude"].isna() | df["Longitude"].isna()
num_cols = [
    "Noofvehicle_involved", "accident_count_6mo", "severity_numeric",
    "road_type_encoded", "road_condition", "junction_control",
    "weather_risk", "hit_run", "main_cause_encoded",
    "road_character_encoded", "is_urban", "is_highway",
    "locality_accident_count", "locality_fatal_count",
    "locality_high_sev_count", "locality_fatal_rate",
    "locality_high_sev_rate", "locality_log_volume",
    "locality_risk_score", "locality_risk_rank",
    "Latitude", "Longitude",
]
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

if "locality" in df.columns:
    df["locality"] = df["locality"].fillna("Unknown").astype(str).str.strip().str.upper()

# ── STEP 4: locality encoding ────────────────────────────────
locality_freq = df["locality"].value_counts()
df["locality_freq_enc"] = df["locality"].map(locality_freq).fillna(1)

locality_risk_mean = df.groupby("locality")[TARGET_COLUMN].mean()
df["locality_risk_enc"] = df["locality"].map(locality_risk_mean).fillna(df[TARGET_COLUMN].mean())

locality_high_risk_rate = df.groupby("locality")[TARGET_COLUMN].apply(
    lambda x: (x == 2).sum() / len(x)
)
df["locality_high_risk_flag"] = df["locality"].map(locality_high_risk_rate).fillna(0)
df["locality_high_risk_flag"] = (df["locality_high_risk_flag"] > 0.4).astype(int)

locality_sev_mean = df.groupby("locality")["severity_numeric"].mean()
df["locality_sev_mean"] = df["locality"].map(locality_sev_mean).fillna(df["severity_numeric"].median())

# ── STEP 5: DBSCAN geo-clustering ────────────────────────────
df["geo_cluster"] = -1
has_real_geo = ~missing_geo_mask
geo_rad = np.radians(df.loc[has_real_geo, ["Latitude", "Longitude"]].values)
eps_rad = 0.5 / 6371.0

db = DBSCAN(eps=eps_rad, min_samples=5, algorithm="ball_tree", metric="haversine")
df.loc[has_real_geo, "geo_cluster"] = db.fit_predict(geo_rad)

# ── STEP 6: feature engineering ──────────────────────────────
df["vehicles_avg"]         = df["Noofvehicle_involved"]
df["log_accident_count"]   = np.log1p(df["accident_count_6mo"])
df["accident_sqrt"]        = np.sqrt(df["accident_count_6mo"])
df["vehicles_log"]         = np.log1p(df["vehicles_avg"])
df["year_recency"]         = df["Year"] - 2016
df["risk_junction"]        = (df["junction_control"] > 0).astype(int)

df["urban_road"]           = df["is_urban"] * df["road_type_encoded"]
df["urban_traffic"]        = df["is_urban"] * df["vehicles_avg"]
df["weather_road"]         = df["weather_risk"] * df["road_condition"]
df["junction_traffic"]     = df["junction_control"] * df["vehicles_avg"]
df["risk_weather_vehicle"] = df["weather_risk"] * df["vehicles_avg"]
df["urban_junction"]       = df["is_urban"] * df["junction_control"]
df["traffic_intensity"]    = df["vehicles_avg"] * df["road_type_encoded"]
df["busy_junction"]        = df["vehicles_avg"] * df["junction_control"]

df["locality_urban_risk"]   = df["locality_risk_enc"] * df["is_urban"]
df["locality_weather_risk"] = df["locality_risk_enc"] * df["weather_risk"]
df["locality_sev_volume"]   = df["locality_sev_mean"] * df["locality_log_volume"]

# ── STEP 7: same dropna using the SAVED feature list ─────────
df = df.dropna(subset=available_features + [TARGET_COLUMN])

# ── STEP 8: identical split ──────────────────────────────────
X = df[available_features]
y = df[TARGET_COLUMN].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"  Test set reproduced: {len(X_test):,} rows")

# ── predict + confusion matrix ───────────────────────────────
y_pred = best_clf.predict(X_test)
labels_order = [0, 1, 2]
cm = confusion_matrix(y_test, y_pred, labels=labels_order)

print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    labels=labels_order,
    target_names=[RISK_LABELS[c] for c in labels_order],
))

print("\nConfusion matrix (rows = actual, cols = predicted):")
header = "            " + "  ".join(f"{RISK_LABELS[c]:>8}" for c in labels_order)
print(header)
for i, row in enumerate(cm):
    print(f"{RISK_LABELS[labels_order[i]]:>10}  " + "  ".join(f"{v:>8}" for v in row))

# Save as JSON
cm_dict = {
    "labels": [RISK_LABELS[c] for c in labels_order],
    "matrix": cm.tolist(),
}
with open(os.path.join(OUTPUT_DIR, "confusion_matrix.json"), "w") as f:
    json.dump(cm_dict, f, indent=2)

# Save as PNG
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[RISK_LABELS[c] for c in labels_order])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
print(f"\n  Saved outputs/confusion_matrix.json")
print(f"  Saved outputs/confusion_matrix.png")