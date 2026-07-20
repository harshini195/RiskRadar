import pandas as pd
import json

# =========================
# LOAD DATASET
# =========================

# Replace with your actual dataset filename
DATASET_PATH = "ml/data/cleaned_accidents_full.csv"

df = pd.read_csv(DATASET_PATH)

# =========================
# CLEAN DATA
# =========================

# Remove rows with missing values
df = df.dropna(
    subset=[
        "Latitude",
        "Longitude",
        "locality",
        "DISTRICTNAME"
    ]
)

# Convert coordinates to numeric
df["Latitude"] = pd.to_numeric(
    df["Latitude"],
    errors="coerce"
)

df["Longitude"] = pd.to_numeric(
    df["Longitude"],
    errors="coerce"
)

# Remove invalid coordinates
df = df.dropna(subset=["Latitude", "Longitude"])

# =========================
# FILTER HIGH RISK ACCIDENTS
# =========================

# If your dataset uses:
# risk_level = High / Medium / Low
print(df["risk_level"].unique())
high_risk = df[
    df["risk_level"] == 2
]

# =========================
# GROUP BY DISTRICT + LOCALITY
# =========================

grouped = high_risk.groupby(
    ["DISTRICTNAME", "locality"]
)

hotspots = []

# =========================
# GENERATE HOTSPOTS
# =========================

for (district, locality), group in grouped:

    # Skip very small accident groups
    if len(group) < 3:
        continue

    hotspot = {

        # District name
        "district": district,

        # Area/locality name
        "locality": locality,

        # Average coordinates
        "lat": round(
            group["Latitude"].mean(),
            6
        ),

        "lng": round(
            group["Longitude"].mean(),
            6
        ),

        # Hotspot severity
        "severity": "high",

        # Number of accidents
        "accidents": int(len(group)),

        # Average hotspot risk score
        "risk_score": round(
            group["locality_risk_score"].mean(),
            2
        ),

        # Fatal accident count
        "fatal_count": int(
            group["locality_fatal_count"].max()
        ),

        # High severity count
        "high_severity_count": int(
            group["locality_high_sev_count"].max()
        )
    }

    hotspots.append(hotspot)

# =========================
# SORT HOTSPOTS
# =========================

hotspots = sorted(
    hotspots,
    key=lambda x: (
        x["risk_score"],
        x["accidents"]
    ),
    reverse=True
)

# =========================
# KEEP TOP HOTSPOTS
# =========================

TOP_N = 200

hotspots = hotspots[:TOP_N]

# =========================
# SAVE JSON
# =========================

OUTPUT_PATH = "outputs/hotspots.json"

with open(OUTPUT_PATH, "w") as f:
    json.dump(hotspots, f, indent=2)

# =========================
# DONE
# =========================

print(f"\nGenerated {len(hotspots)} hotspots")
print(f"Saved to: {OUTPUT_PATH}")