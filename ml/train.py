"""
Accident Risk Prediction - train.py
Dataset : cleaned_accidents.csv
Target  : risk_level  (0=Low, 1=Medium, 2=High)

Locality is treated as a FIRST-CLASS feature:
  - Frequency encoded (how many accidents in that locality)
  - Risk encoded (mean risk_level per locality)
  - All pre-computed locality aggregates used directly
  - DBSCAN geo-cluster on Lat/Lon adds spatial context
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from sklearn.cluster import DBSCAN
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH = "data/cleaned_accidents_full.csv"
TARGET_COLUMN = "risk_level"
OUTPUT_DIR    = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}

# ─────────────────────────────────────────────
# STEP 1 — Load
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data...")
print("=" * 60)
df = pd.read_csv(DATA_PATH, encoding="latin1")
print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")

# ─────────────────────────────────────────────
# STEP 2 — Filter valid targets
# ─────────────────────────────────────────────
print("\nSTEP 2: Filtering valid risk_level rows...")
before = len(df)
df = df[df[TARGET_COLUMN].isin([0, 1, 2])].copy()
print(f"  Kept {len(df):,} rows  (dropped {before - len(df):,})")

# ─────────────────────────────────────────────
# STEP 3 — Imputation
# ─────────────────────────────────────────────
print("\nSTEP 3: Imputing missing values...")
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

# ─────────────────────────────────────────────
# STEP 4 — LOCALITY ENCODING (most important)
# ─────────────────────────────────────────────
print("\nSTEP 4: Encoding locality as primary risk signal...")

# 4a. Frequency encoding — how busy is this locality?
locality_freq = df["locality"].value_counts()
df["locality_freq_enc"] = df["locality"].map(locality_freq).fillna(1)

# 4b. Risk encoding — mean risk_level per locality (target encoding)
locality_risk_mean = df.groupby("locality")[TARGET_COLUMN].mean()
df["locality_risk_enc"] = df["locality"].map(locality_risk_mean).fillna(df[TARGET_COLUMN].mean())

# 4c. High-risk flag — localities where >40% accidents are high risk
locality_high_risk_rate = df.groupby("locality")[TARGET_COLUMN].apply(
    lambda x: (x == 2).sum() / len(x)
)
df["locality_high_risk_flag"] = df["locality"].map(locality_high_risk_rate).fillna(0)
df["locality_high_risk_flag"] = (df["locality_high_risk_flag"] > 0.4).astype(int)

# 4d. Locality severity mean
locality_sev_mean = df.groupby("locality")["severity_numeric"].mean()
df["locality_sev_mean"] = df["locality"].map(locality_sev_mean).fillna(df["severity_numeric"].median())

# Save locality encodings for inference
locality_encodings = pd.DataFrame({
    "locality":                locality_freq.index,
    "locality_freq_enc":       locality_freq.values,
    "locality_risk_enc":       locality_freq.index.map(locality_risk_mean),
    "locality_high_risk_flag": ((locality_freq.index.map(locality_high_risk_rate) > 0.4).astype(int)).tolist(),
    "locality_sev_mean":       locality_freq.index.map(locality_sev_mean),
})
locality_encodings.to_json(
    os.path.join(OUTPUT_DIR, "locality_encodings.json"),
    orient="records", indent=2
)
print(f"  locality_freq_enc     : {df['locality_freq_enc'].nunique()} unique values")
print(f"  locality_risk_enc     : min={df['locality_risk_enc'].min():.3f}  max={df['locality_risk_enc'].max():.3f}")
print(f"  locality_high_risk_flag=1 : {df['locality_high_risk_flag'].sum():,} rows")
print(f"  Saved locality_encodings.json  ({locality_encodings.shape[0]} localities)")

# ─────────────────────────────────────────────
# STEP 5 — DBSCAN geo-clustering
# ─────────────────────────────────────────────
print("\nSTEP 5: DBSCAN geo-clustering (Latitude / Longitude)...")

geo_rad = np.radians(df[["Latitude", "Longitude"]].values)
eps_rad = 0.5 / 6371.0   # 500 m in radians

db = DBSCAN(eps=eps_rad, min_samples=5, algorithm="ball_tree", metric="haversine")
df["geo_cluster"] = db.fit_predict(geo_rad)

n_clusters = len(set(df["geo_cluster"])) - (1 if -1 in df["geo_cluster"].values else 0)
noise_pts   = (df["geo_cluster"] == -1).sum()
print(f"  Clusters found : {n_clusters}")
print(f"  Noise points   : {noise_pts:,}  (geo_cluster = -1)")

# Save locality → geo_cluster map for inference
cluster_map = (
    df.groupby("locality")["geo_cluster"]
    .agg(lambda x: int(x.mode()[0]))
    .to_dict()
)
with open(os.path.join(OUTPUT_DIR, "locality_cluster_map.json"), "w") as f:
    json.dump({str(k): v for k, v in cluster_map.items()}, f, indent=2)
print(f"  Saved locality_cluster_map.json")

# ─────────────────────────────────────────────
# STEP 6 — Feature engineering
# ─────────────────────────────────────────────
print("\nSTEP 6: Engineering additional features...")

df["vehicles_avg"]         = df["Noofvehicle_involved"]
df["log_accident_count"]   = np.log1p(df["accident_count_6mo"])
df["accident_sqrt"]        = np.sqrt(df["accident_count_6mo"])
df["vehicles_log"]         = np.log1p(df["vehicles_avg"])
df["year_recency"]         = df["Year"] - 2016
df["risk_junction"]        = (df["junction_control"] > 0).astype(int)

# Interaction features
df["urban_road"]           = df["is_urban"] * df["road_type_encoded"]
df["urban_traffic"]        = df["is_urban"] * df["vehicles_avg"]
df["weather_road"]         = df["weather_risk"] * df["road_condition"]
df["junction_traffic"]     = df["junction_control"] * df["vehicles_avg"]
df["risk_weather_vehicle"] = df["weather_risk"] * df["vehicles_avg"]
df["urban_junction"]       = df["is_urban"] * df["junction_control"]
df["traffic_intensity"]    = df["vehicles_avg"] * df["road_type_encoded"]
df["busy_junction"]        = df["vehicles_avg"] * df["junction_control"]

# Locality x other interactions — captures locality-specific risk amplifiers
df["locality_urban_risk"]   = df["locality_risk_enc"] * df["is_urban"]
df["locality_weather_risk"] = df["locality_risk_enc"] * df["weather_risk"]
df["locality_sev_volume"]   = df["locality_sev_mean"] * df["locality_log_volume"]

# ─────────────────────────────────────────────
# STEP 7 — Final feature list
# ─────────────────────────────────────────────
FEATURE_COLUMNS = [
    # Locality (safe)
    "locality_freq_enc",
    "locality_accident_count",
    "locality_log_volume",

    # Geo
    "geo_cluster",

    # Core features
    "road_type_encoded",
    "road_condition",
    "junction_control",
    "weather_risk",
    "hit_run",
    "main_cause_encoded",
    "road_character_encoded",
    "is_urban",
    "is_highway",

    # Vehicle / traffic
    "vehicles_avg",
    "log_accident_count",
    "accident_sqrt",
    "vehicles_log",

    # Time
    "year_recency",

    # Interactions (SAFE ones only)
    "urban_road",
    "risk_junction",
    "urban_traffic",
    "weather_road",
    "junction_traffic",
    "risk_weather_vehicle",
    "urban_junction",
    "traffic_intensity",
    "busy_junction",
]

available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
missing_cols       = [col for col in FEATURE_COLUMNS if col not in df.columns]
if missing_cols:
    print(f"  ⚠  Skipped (not in df): {missing_cols}")

df = df.dropna(subset=available_features + [TARGET_COLUMN])
print(f"\n  Final rows        : {len(df):,}")
print(f"  Features used     : {len(available_features)}")
print(f"  Class distribution:\n{df[TARGET_COLUMN].value_counts().sort_index().to_string()}")

# ─────────────────────────────────────────────
# STEP 8 — Train / Test split
# ─────────────────────────────────────────────
print("\nSTEP 8: Splitting 70 / 30 ...")
X = df[available_features]
y = df[TARGET_COLUMN].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"  Train : {len(X_train):,}  |  Test : {len(X_test):,}")

# ─────────────────────────────────────────────
# STEP 9 — Models
# ─────────────────────────────────────────────
print("\nSTEP 9: Training models...")

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=5,
        min_samples_leaf=5,
        subsample=0.9,
        random_state=42,
    ),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced",
            C=0.5, random_state=42,
        )),
    ]),
}

results    = {}
best_model = None
best_f1    = 0.0

for name, model in models.items():
    print(f"\n  ── {name} ──")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1     = f1_score(y_test, y_pred, average="weighted")

    report = classification_report(
        y_test, y_pred,
        target_names=[RISK_LABELS[i] for i in range(3)],
        output_dict=True,
    )
    results[name] = {
        "f1_weighted": round(f1, 4),
        "per_class": {
            RISK_LABELS[i]: {
                "precision": round(report[RISK_LABELS[i]]["precision"], 4),
                "recall":    round(report[RISK_LABELS[i]]["recall"], 4),
                "f1":        round(report[RISK_LABELS[i]]["f1-score"], 4),
            }
            for i in range(3)
        },
    }
    print(f"  F1 (weighted): {f1:.4f}")
    print(classification_report(
        y_test, y_pred,
        target_names=[RISK_LABELS[i] for i in range(3)],
    ))
    if f1 > best_f1:
        best_f1    = f1
        best_model = (name, model)

# ─────────────────────────────────────────────
# STEP 10 — Feature importance
# ─────────────────────────────────────────────
print("\nSTEP 10: Feature importance...")
best_name, best_clf = best_model
print(f"  Best : {best_name}  (F1 = {best_f1:.4f})")

clf_inner = best_clf.named_steps["clf"] if hasattr(best_clf, "named_steps") else best_clf
if hasattr(clf_inner, "feature_importances_"):
    feat_imp = sorted(
        zip(available_features, clf_inner.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("\n  Top 25 Feature Importances:")
    for feat, imp in feat_imp[:25]:
        bar = "█" * int(imp * 50)
        print(f"    {feat:<38} {imp:.4f}  {bar}")
    results["feature_importance"] = {f: round(float(i), 4) for f, i in feat_imp}

# ─────────────────────────────────────────────
# STEP 11 — Hotspot reports
# ─────────────────────────────────────────────
print("\nSTEP 11: Hotspot reports...")
df_full = pd.read_csv(DATA_PATH, encoding="latin1")
df_full = df_full[df_full[TARGET_COLUMN].isin([0, 1, 2])]
if "locality" in df_full.columns:
    df_full["locality"] = df_full["locality"].fillna("Unknown").astype(str).str.strip().str.upper()

# District-level
district_hs = (
    df_full.groupby("DISTRICTNAME")
    .agg(
        total_accidents   =("risk_level", "count"),
        high_risk_count   =("risk_level", lambda x: (x == 2).sum()),
        medium_risk_count =("risk_level", lambda x: (x == 1).sum()),
    )
    .assign(
        high_risk_rate  =lambda d: (d["high_risk_count"] / d["total_accidents"] * 100).round(2),
        medium_risk_rate=lambda d: (d["medium_risk_count"] / d["total_accidents"] * 100).round(2),
    )
    .sort_values("high_risk_rate", ascending=False)
    .head(15).reset_index()
)
district_hs.to_json(os.path.join(OUTPUT_DIR, "hotspots.json"), orient="records", indent=2)

# Locality-level — granular, most actionable
locality_hs = (
    df_full.groupby(["locality", "DISTRICTNAME"])
    .agg(
        total      =("risk_level", "count"),
        high_risk  =("risk_level", lambda x: (x == 2).sum()),
        medium_risk=("risk_level", lambda x: (x == 1).sum()),
    )
    .assign(high_risk_rate=lambda d: (d["high_risk"] / d["total"] * 100).round(2))
    .sort_values("high_risk_rate", ascending=False)
    .head(50).reset_index()
)
locality_hs.to_json(
    os.path.join(OUTPUT_DIR, "locality_hotspots.json"), orient="records", indent=2
)
print(f"  Top 5 high-risk localities:")
print(locality_hs[["locality", "DISTRICTNAME", "total", "high_risk", "high_risk_rate"]].head(5).to_string(index=False))

# Accident_Location-level
if "Accident_Location" in df_full.columns:
    location_hs = (
        df_full.groupby(["DISTRICTNAME", "Accident_Location"])
        .agg(
            total     =("risk_level", "count"),
            high_risk =("risk_level", lambda x: (x == 2).sum()),
        )
        .assign(high_risk_rate=lambda d: (d["high_risk"] / d["total"] * 100).round(2))
        .sort_values("high_risk_rate", ascending=False)
        .head(30).reset_index()
    )
    location_hs.to_json(
        os.path.join(OUTPUT_DIR, "location_hotspots.json"), orient="records", indent=2
    )

# ─────────────────────────────────────────────
# STEP 12 — Save artefacts
# ─────────────────────────────────────────────
print("\nSTEP 12: Saving artefacts...")
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)
with open(os.path.join(OUTPUT_DIR, "best_model.pkl"), "wb") as f:
    pickle.dump(best_clf, f)
with open(os.path.join(OUTPUT_DIR, "feature_columns.pkl"), "wb") as f:
    pickle.dump(available_features, f)

print("  ✓  best_model.pkl")
print("  ✓  metrics.json")
print("  ✓  feature_columns.pkl")
print("  ✓  locality_encodings.json")
print("  ✓  locality_cluster_map.json")
print("  ✓  hotspots.json")
print("  ✓  locality_hotspots.json")
print("  ✓  location_hotspots.json")

print("\n" + "=" * 60)
print(f"  DONE — Best: {best_name} | F1: {best_f1:.4f}")
print("=" * 60)