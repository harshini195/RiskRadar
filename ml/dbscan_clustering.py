"""
DBSCAN Clustering for Accident Hotspot Detection
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Paths (robust way)
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_accidents.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "dbscan_output.csv")

# -------------------------------
# Load Data
# -------------------------------
print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH, encoding="latin1")
print(f"Loaded {len(df):,} rows")

# -------------------------------
# Feature Engineering (IMPORTANT)
# -------------------------------
print("\nApplying feature engineering...")

# Fill missing values
df["accident_count_6mo"] = df["accident_count_6mo"].fillna(df["accident_count_6mo"].median())

# Create derived features (same as train.py)
df["log_accident_count"] = np.log1p(df["accident_count_6mo"])
df["year_recency"] = df["Year"] - 2016
df["urban_road"] = df["is_urban"] * df["road_type_encoded"]
df["risk_junction"] = (df["junction_control"] > 0).astype(int)

# -------------------------------
# Select Features (NO TARGET)
# -------------------------------
features = [
    "log_accident_count",
    "vehicles_avg",
    "road_type_encoded",
    "road_condition",
    "weather_risk",
    "is_urban"
]

# Check if all features exist
missing = [col for col in features if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

X = df[features].fillna(0)

# -------------------------------
# Scaling
# -------------------------------
print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Apply DBSCAN
# -------------------------------
print("\nRunning DBSCAN clustering...")

db = DBSCAN(eps=0.8, min_samples=10)
clusters = db.fit_predict(X_scaled)

# -------------------------------
# Add Results
# -------------------------------
df["cluster"] = clusters

# -------------------------------
# Save Output
# -------------------------------
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved clustered data to: {OUTPUT_PATH}")

# -------------------------------
# Summary
# -------------------------------
print("\nCluster Summary:")
print("Clusters found:", set(clusters))
print(df["cluster"].value_counts())

# -------------------------------
# Hotspot Extraction
# -------------------------------
hotspots = df[df["cluster"] != -1]
print(f"\nHotspot points: {len(hotspots):,}")
print(f"Noise points: {(df['cluster'] == -1).sum():,}")

print("\nTop clusters (excluding noise):")
print(hotspots["cluster"].value_counts().head())