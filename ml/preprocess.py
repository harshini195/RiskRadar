"""
RiskRadar — Complete Data Preprocessing Pipeline
==================================================
Dataset : AccidentReports.csv (Karnataka Police Records)
Run     : python3 preprocess.py
Output  : ml/data/cleaned_accidents.csv + preprocessing_report.txt
"""

import pandas as pd
import numpy as np
import re
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'AccidentReports.csv')
OUT_PATH  = os.path.join(BASE_DIR, 'data', 'cleaned_accidents.csv')
REPORT    = os.path.join(BASE_DIR, 'data', 'preprocessing_report.txt')

lines = []
def log(msg=''):
    print(msg)
    lines.append(str(msg))

log("=" * 60)
log("  RISKRADAR — DATA PREPROCESSING PIPELINE")
log("=" * 60)

# ── STEP 1: LOAD ──────────────────────────────────────────────────────────────
log("\n── STEP 1: LOAD RAW DATA")
df = pd.read_csv(DATA_PATH, encoding='latin1')
original_shape = df.shape
log(f"Loaded : {len(df):,} rows x {len(df.columns)} columns")

# ── STEP 2: DROP USELESS COLUMNS ─────────────────────────────────────────────
log("\n── STEP 2: DROP IRRELEVANT / HIGH-NULL COLUMNS")
drop_cols = [
    'RoadJunction', 'Distance_LandMark_Second', 'Road_Markings',
    'landmark_second', 'Side_Walk', 'Spot_Conditions', 'Lane_Type',
    'Accident_SpotB', 'Collision_TypeB', 'Crime_No', 'RI',
    'Landmark_first', 'Distance_LandMark_First', 'Accident_Description',
    'Accident_Road', 'Accident_SubLocation',
]
dropped = [c for c in drop_cols if c in df.columns]
df.drop(columns=dropped, inplace=True)
log(f"Dropped {len(dropped)} columns -> {len(df.columns)} remaining")

# ── STEP 3: EXTRACT LOCALITY FROM UNITNAME ───────────────────────────────────
log("\n── STEP 3: EXTRACT LOCALITY FROM UNITNAME")
log("  Stripping police-unit suffixes (Traffic PS, Rural PS, PS, etc.)")

def extract_locality(name):
    """
    Strip police-station suffixes from UNITNAME to get the actual locality.

    Examples:
      'Nelamangala Traffic PS'       -> 'Nelamangala'
      'Amengad PS'                   -> 'Amengad'
      'Arasikere Rural PS'           -> 'Arasikere'
      'Kalaburagi Traffic II PS'     -> 'Kalaburagi'
      'Belgaum North Traffic PS'     -> 'Belgaum North'
      'Davanagere South Traffic PS'  -> 'Davanagere South'
    """
    name = re.sub(r'\s+', ' ', str(name).strip())

    # Ordered most-specific -> least-specific so longer patterns match first
    patterns = [
        r'\s+Traffic\s+(I{1,3}|IV)\s*PS$',   # Traffic I / II / III / IV PS
        r'\s+Traffic\s+PS$',                  # Traffic PS
        r'\s+Traffic\s+Police\s+Station$',    # Traffic Police Station
        r'\s+Rural\s+PS$',                    # Rural PS
        r'\s+Town\s+PS$',                     # Town PS
        r'\s+Rly\s+PS$',                      # Railway PS
        r'\s+North\s+Traffic\s+PS$',          # North Traffic PS
        r'\s+South\s+Traffic\s+PS$',          # South Traffic PS
        r'\s+East\s+Traffic\s+PS$',           # East Traffic PS
        r'\s+West\s+Traffic\s+PS$',           # West Traffic PS
        r'\s+North\s+PS$',                    # North PS
        r'\s+South\s+PS$',                    # South PS
        r'\s+East\s+PS$',                     # East PS
        r'\s+West\s+PS$',                     # West PS
        r'\s+City\s+PS$',                     # City PS
        r'\s+Police\s+Station$',              # Police Station
        r'\s+PS$',                            # plain PS — catch-all (LAST)
    ]
    for pat in patterns:
        cleaned = re.sub(pat, '', name, flags=re.IGNORECASE).strip()
        if cleaned != name:
            return cleaned
    return name   # no suffix matched — return as-is

df['locality'] = df['UNITNAME'].apply(extract_locality)

n_unique = df['locality'].nunique()
log(f"  Extracted {n_unique} unique localities from {df['UNITNAME'].nunique()} unit names")
log(f"  Sample transformations:")
sample_pairs = (
    df[['UNITNAME', 'locality']]
    .drop_duplicates()
    .sample(10, random_state=42)
)
for _, row in sample_pairs.iterrows():
    log(f"    '{row['UNITNAME']}'  ->  '{row['locality']}'")

# ── STEP 4: REMOVE DUPLICATES ─────────────────────────────────────────────────
log("\n── STEP 4: REMOVE DUPLICATES")
before = len(df)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
log(f"Removed {before - len(df):,} duplicate rows -> {len(df):,} remaining")

# ── STEP 5: CLEAN SEVERITY ────────────────────────────────────────────────────
log("\n── STEP 5: CLEAN SEVERITY COLUMN")
valid_severity = ['Grievous Injury', 'Simple Injury', 'Fatal', 'Damage Only', 'Not Applicable']
before = len(df)
df = df[df['Severity'].isin(valid_severity)].reset_index(drop=True)
log(f"Removed {before - len(df):,} rows with invalid Severity values")
log(f"Severity distribution:\n{df['Severity'].value_counts().to_string()}")

# ── STEP 6: FIX COORDINATES ───────────────────────────────────────────────────
log("\n── STEP 6: FIX COORDINATES")
KARNATAKA_LAT = (11.5, 18.5)
KARNATAKA_LON = (74.0, 78.6)

valid_geo = (
    df['Latitude'].between(*KARNATAKA_LAT) &
    df['Longitude'].between(*KARNATAKA_LON)
)
invalid_count = (~valid_geo).sum()
log(f"Rows with invalid coordinates: {invalid_count:,} ({invalid_count / len(df) * 100:.1f}%)")
df.loc[~valid_geo, ['Latitude', 'Longitude']] = np.nan

DISTRICT_CENTROIDS = {
    'Bengaluru City':   (12.9716, 77.5946),
    'Bengaluru Dist':   (13.0827, 77.5877),
    'Tumakuru':         (13.3409, 77.1010),
    'Hassan':           (13.0033, 76.1004),
    'Mandya':           (12.5218, 76.8951),
    'Belagavi Dist':    (15.8497, 74.4977),
    'Mysuru Dist':      (12.2958, 76.6394),
    'Shivamogga':       (13.9299, 75.5681),
    'Chitradurga':      (14.2251, 76.3980),
    'Ramanagara':       (12.7157, 77.2822),
    'Udupi':            (13.3409, 74.7421),
    'Uttara Kannada':   (14.7937, 74.6775),
    'Bidar':            (17.9140, 77.5199),
    'Mangaluru City':   (12.9141, 74.8560),
    'Davanagere':       (14.4644, 75.9218),
    'Kalaburagi':       (17.3297, 76.8200),
    'Ballari':          (15.1394, 76.9214),
    'Vijayapura':       (16.8302, 75.7100),
    'Dharwad':          (15.4589, 75.0078),
    'Gadag':            (15.4315, 75.6215),
    'Haveri':           (14.7957, 75.3991),
    'Koppal':           (15.3508, 76.1549),
    'Raichur':          (16.2120, 77.3566),
    'Yadgir':           (16.7710, 77.1384),
    'Bagalkot':         (16.1826, 75.6961),
    'Kodagu':           (12.3375, 75.8069),
    'Chikkamagaluru':   (13.3161, 75.7720),
    'Tumkur':           (13.3409, 77.1010),
    'Dakshina Kannada': (12.8438, 75.2479),
    'Chamarajanagar':   (11.9216, 76.9395),
    'Chikkaballapur':   (13.4355, 77.7280),
    'Kolar':            (13.1368, 78.1297),
    'Ramnagara':        (12.7157, 77.2822),
    'Vijayanagara':     (15.1394, 76.9214),
    'Yadagiri':         (16.7710, 77.1384),
}

np.random.seed(42)
filled = 0
for idx, row in df[df['Latitude'].isna()].iterrows():
    centroid = DISTRICT_CENTROIDS.get(row['DISTRICTNAME'])
    if centroid:
        df.at[idx, 'Latitude']  = centroid[0] + np.random.uniform(-0.15, 0.15)
        df.at[idx, 'Longitude'] = centroid[1] + np.random.uniform(-0.15, 0.15)
        filled += 1

log(f"Filled {filled:,} missing coordinates with district centroids (+ jitter)")
log(f"Remaining coordinate nulls: {df['Latitude'].isna().sum():,}")

# ── STEP 7: HANDLE OUTLIERS ───────────────────────────────────────────────────
log("\n── STEP 7: HANDLE OUTLIERS")
df['Noofvehicle_involved'] = df['Noofvehicle_involved'].clip(1, 10)
log("Vehicles involved capped at 10")

# ── STEP 8: FILL REMAINING MISSING VALUES ────────────────────────────────────
log("\n── STEP 8: FILL REMAINING MISSING VALUES")
for col in df.select_dtypes(include='object').columns:
    null_count = df[col].isnull().sum()
    if null_count > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        log(f"  {col}: filled {null_count:,} nulls with mode='{mode_val}'")
for col in df.select_dtypes(include=[np.number]).columns:
    null_count = df[col].isnull().sum()
    if null_count > 0 and col not in ['Latitude', 'Longitude']:
        df[col].fillna(df[col].median(), inplace=True)
        log(f"  {col}: filled {null_count:,} nulls with median")
log(f"Total remaining nulls: {df.isnull().sum().sum()}")

# ── STEP 9: ENCODE SEVERITY ───────────────────────────────────────────────────
log("\n── STEP 9: ENCODE SEVERITY -> RISK LEVEL")
severity_map = {
    'Fatal':           3,
    'Grievous Injury': 2,
    'Simple Injury':   1,
    'Damage Only':     0,
    'Not Applicable':  0,
}
df['severity_numeric'] = df['Severity'].map(severity_map)
df['risk_level'] = df['severity_numeric'].apply(
    lambda x: 2 if x == 3 else 1 if x == 2 else 0
)
log(f"Risk level distribution:\n{df['risk_level'].value_counts().sort_index().to_string()}")

# ── STEP 10: ENCODE CATEGORICALS ─────────────────────────────────────────────
log("\n── STEP 10: ENCODE CATEGORICAL FEATURES")

road_type_map = {
    'Expressway': 5, 'NH': 4, 'State Highway': 3,
    'Major District Road': 2, 'Minor District Road': 2,
    'Arterial': 2, 'Sub Arterial': 2, 'Two way': 2,
    'City or Town Road': 1, 'Residential Street': 1,
    'Service Road': 1, 'One way': 1, 'Village Road': 0,
    'Forest Road': 0, 'Mixed': 1, 'Feeder Road': 0,
    'Not Applicable': 1, 'Others': 1,
}
df['road_type_encoded'] = df['Road_Type'].map(road_type_map).fillna(1).astype(int)

surface_map = {
    'Dry': 2, 'Fine': 2, 'Wet': 1, 'Others': 1,
    'Not Applicable': 1, 'Muddy': 0,
    'Ditch or Potholed': 0, 'Flooded': 0,
}
df['road_condition'] = df['Surface_Condition'].map(surface_map).fillna(1).astype(int)

junction_map = {
    'Not at  Junction': 0, 'Not Applicable': 0,
    'Uncontrolled': 1, 'No signal lights': 1, 'Give way sign': 1,
    'Controlled': 2, 'Stop sign': 2, 'Stop Sign': 2,
    'Signal lights Automatic': 3, 'Signal lights Blinking': 3,
    'Signals (Working)': 3, 'Signal lights Manual': 3,
    'Signal lights Not working': 1, 'Signals (Not working)': 1,
    'Police / Manual': 2, 'T Junction': 1,
}
df['junction_control'] = df['Junction_Control'].map(junction_map).fillna(0).astype(int)

df['weather_risk'] = df['Weather'].apply(
    lambda x: 1 if any(w in str(x).lower()
        for w in ['rain', 'fog', 'mist', 'snow', 'flood', 'wind', 'hail', 'dust', 'cold']) else 0
)

df['hit_run'] = (df['Hit_Run'] == 'Yes').astype(int)

cause_map = {
    'Human Error': 2, 'Vehicle Defect': 1,
    'Road Environment Defect': 1, 'Not Applicable': 0,
}
df['main_cause_encoded'] = df['Main_Cause'].map(cause_map).fillna(0).astype(int)

df['is_highway'] = df['Accident_Spot'].apply(
    lambda x: 1 if any(w in str(x).lower()
        for w in ['highway', 'nh', 'sh', 'national']) else 0
)

road_char_map = {
    'Straight Road': 0, 'Straight and flat': 0,
    'Slight Curve': 1, 'Curve': 1, 'Curved': 1,
    'Incline': 1, 'Gentle Incline or Climb': 1, 'Dip or trough': 1,
    'Hump': 1, 'Crest of hill': 2, 'Sharp Curve': 2,
    'Curve and Incline': 2, 'Steep Incline or Climb': 2,
    'Bridge': 2, 'Culvert': 2,
    'Not Applicable': 0, 'Others': 0,
}
df['road_character_encoded'] = df['Road_Character'].map(road_char_map).fillna(0).astype(int)

df['is_urban'] = df['Accident_Location'].apply(
    lambda x: 1 if any(w in str(x).lower()
        for w in ['city', 'town', 'urban', 'municipal', 'market', 'school', 'hospital']) else 0
)

log("Encoded: road_type, road_condition, junction_control, weather_risk,")
log("         hit_run, main_cause, is_highway, road_character, is_urban")

# ── STEP 11: LOCALITY-LEVEL RISK FEATURES ────────────────────────────────────
log("\n── STEP 11: LOCALITY-LEVEL RISK FEATURES (from UNITNAME)")
log("  Computing per-locality accident volume, fatal rate, and high-severity rate...")

locality_stats = (
    df.assign(
        _is_fatal=df['Severity'].eq('Fatal'),
        _is_high_sev=df['Severity'].isin(['Fatal', 'Grievous Injury']),
    )
    .groupby('locality')
    .agg(
        locality_accident_count=('Severity', 'count'),
        locality_fatal_count=('_is_fatal', 'sum'),
        locality_high_sev_count=('_is_high_sev', 'sum'),
    )
    .assign(
        # Fraction of accidents at this locality that end in death
        locality_fatal_rate=lambda d: (
            d['locality_fatal_count'] / d['locality_accident_count']
        ).round(4),
        # Fraction that are fatal OR grievous injury
        locality_high_sev_rate=lambda d: (
            d['locality_high_sev_count'] / d['locality_accident_count']
        ).round(4),
        # Log-scaled volume (reduces outlier skew from very busy localities)
        locality_log_volume=lambda d: np.log1p(d['locality_accident_count']).round(4),
        # Composite risk score = log-volume × weighted severity
        # A high-fatal-rate locality with many accidents scores highest
        locality_risk_score=lambda d: (
            d['locality_log_volume'] *
            (0.6 * d['locality_fatal_rate'] + 0.4 * d['locality_high_sev_rate'])
        ).round(4),
    )
    .reset_index()
)

# 0-1 percentile rank — 1.0 = most dangerous locality in Karnataka
locality_stats['locality_risk_rank'] = (
    locality_stats['locality_risk_score'].rank(pct=True).round(4)
)

df = df.merge(locality_stats, on='locality', how='left')

log(f"  Added 6 locality-level columns for {locality_stats.shape[0]} unique localities")
log(f"\n  Top 10 most dangerous localities (composite risk score):")
top10 = locality_stats.nlargest(10, 'locality_risk_score')[
    ['locality', 'locality_accident_count', 'locality_fatal_rate',
     'locality_high_sev_rate', 'locality_risk_score', 'locality_risk_rank']
]
log(top10.to_string(index=False))

log(f"\n  Top 10 highest fatal rate localities (min 100 accidents):")
top_fatal = (
    locality_stats[locality_stats['locality_accident_count'] >= 100]
    .nlargest(10, 'locality_fatal_rate')
    [['locality', 'locality_accident_count', 'locality_fatal_rate', 'locality_risk_rank']]
)
log(top_fatal.to_string(index=False))

# ── STEP 12: ACCIDENT COUNT PROXY ────────────────────────────────────────────
log("\n── STEP 12: ENGINEER ACCIDENT COUNT PROXY")
district_road_counts = (
    df.groupby(['DISTRICTNAME', 'Road_Type'])
    .size()
    .reset_index(name='accident_count_6mo')
)
df = df.merge(district_road_counts, on=['DISTRICTNAME', 'Road_Type'], how='left')
df['accident_count_6mo'] = df['accident_count_6mo'].clip(0, 500)
log(f"accident_count_6mo range: {df['accident_count_6mo'].min()} - {df['accident_count_6mo'].max()}")

# ── STEP 13: HANDLE CLASS IMBALANCE ──────────────────────────────────────────
log("\n── STEP 13: HANDLE CLASS IMBALANCE")
class_counts = df['risk_level'].value_counts()
log(f"Before balancing:\n{class_counts.to_string()}")
min_class_size = class_counts.min()
target_size = min(min_class_size * 3, class_counts.max())
balanced_dfs = []
for label in sorted(df['risk_level'].unique()):
    class_df = df[df['risk_level'] == label]
    if len(class_df) > target_size:
        class_df = class_df.sample(n=target_size, random_state=42)
    balanced_dfs.append(class_df)
df_balanced = (
    pd.concat(balanced_dfs)
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)
log(f"After balancing:\n{df_balanced['risk_level'].value_counts().sort_index().to_string()}")

# ── STEP 14: SELECT FINAL FEATURES ───────────────────────────────────────────
log("\n── STEP 14: SELECT FINAL FEATURES")
ML_FEATURES = [
    # Road & environment features
    'accident_count_6mo', 'severity_numeric', 'road_type_encoded',
    'road_condition', 'junction_control', 'weather_risk',
    'Noofvehicle_involved', 'main_cause_encoded', 'hit_run',
    'is_highway', 'road_character_encoded', 'is_urban',
    # Locality-level risk features (NEW — from UNITNAME)
    'locality_accident_count',   # raw accident volume at this locality
    'locality_fatal_rate',       # fraction of accidents that are fatal
    'locality_high_sev_rate',    # fraction fatal OR grievous
    'locality_log_volume',       # log-scaled volume
    'locality_risk_score',       # composite: volume x severity
    'locality_risk_rank',        # 0-1 percentile rank (best single ML feature)
    # Target + metadata
    'risk_level', 'Latitude', 'Longitude',
    'DISTRICTNAME', 'locality', 'Year',
]
df_ml = df_balanced[ML_FEATURES].copy()
df_ml.rename(columns={'Noofvehicle_involved': 'vehicles_avg'}, inplace=True)
log(f"Final dataset: {df_ml.shape[0]:,} rows x {df_ml.shape[1]} columns")

# ── STEP 15: SAVE ─────────────────────────────────────────────────────────────
log("\n── STEP 15: SAVE")
df_ml.to_csv(OUT_PATH, index=False)
log(f"Saved ML dataset  -> {OUT_PATH}")
full_out = os.path.join(BASE_DIR, 'data', 'cleaned_accidents_full.csv')
df.to_csv(full_out, index=False)
log(f"Saved full dataset -> {full_out}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("PREPROCESSING COMPLETE")
log("=" * 60)
log(f"Original  : {original_shape[0]:,} rows x {original_shape[1]} cols")
log(f"ML dataset: {df_ml.shape[0]:,} rows x {df_ml.shape[1]} cols")
log(f"""
New locality features (from UNITNAME):
  locality               -> cleaned area name e.g. 'Nelamangala', 'Amengad'
  locality_accident_count-> total accidents from this locality
  locality_fatal_rate    -> deaths / total at this locality
  locality_high_sev_rate -> (fatal + grievous) / total
  locality_log_volume    -> log(1 + count) — volume without outlier skew
  locality_risk_score    -> composite risk (volume x severity weights)
  locality_risk_rank     -> 0-1 percentile rank — best locality ML signal

Also update FEATURE_COLUMNS in train.py to include:
  'locality_fatal_rate', 'locality_high_sev_rate',
  'locality_log_volume', 'locality_risk_score', 'locality_risk_rank'
""")
log(f"Next step : python3 ml/train.py")

with open(REPORT, 'w') as f:
    f.write('\n'.join(lines))
log(f"Report saved -> {REPORT}")