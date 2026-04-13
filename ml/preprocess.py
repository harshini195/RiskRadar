"""
RiskRadar — Complete Data Preprocessing Pipeline
==================================================
Dataset: AccidentReports.csv (Karnataka Police Records)
Run: python3 preprocess.py
Output: ml/data/cleaned_accidents.csv + preprocessing_report.txt
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'AccidentReports.csv')
OUT_PATH   = os.path.join(BASE_DIR, 'data', 'cleaned_accidents.csv')
REPORT     = os.path.join(BASE_DIR, 'data', 'preprocessing_report.txt')

lines = []
def log(msg=''):
    print(msg)
    lines.append(str(msg))

log("=" * 60)
log("  RISKRADAR — DATA PREPROCESSING PIPELINE")
log("=" * 60)

# STEP 1 — LOAD
log("\n── STEP 1: LOAD RAW DATA")
df = pd.read_csv(DATA_PATH, encoding='latin1')
original_shape = df.shape
log(f"Loaded : {len(df):,} rows x {len(df.columns)} columns")

# STEP 2 — DROP USELESS COLUMNS
log("\n── STEP 2: DROP IRRELEVANT / HIGH-NULL COLUMNS")
drop_cols = [
    'RoadJunction', 'Distance_LandMark_Second', 'Road_Markings',
    'landmark_second', 'Side_Walk', 'Spot_Conditions', 'Lane_Type',
    'Accident_SpotB', 'Collision_TypeB', 'Crime_No', 'RI',
    'Landmark_first', 'Distance_LandMark_First', 'Accident_Description',
    'Accident_Road', 'Accident_SubLocation',
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
log(f"Dropped {len(drop_cols)} columns -> {len(df.columns)} remaining")

# STEP 3 — REMOVE DUPLICATES
log("\n── STEP 3: REMOVE DUPLICATES")
before = len(df)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
log(f"Removed {before - len(df):,} duplicate rows -> {len(df):,} remaining")

# STEP 4 — CLEAN SEVERITY
log("\n── STEP 4: CLEAN SEVERITY COLUMN")
valid_severity = ['Grievous Injury', 'Simple Injury', 'Fatal', 'Damage Only', 'Not Applicable']
before = len(df)
df = df[df['Severity'].isin(valid_severity)]
df.reset_index(drop=True, inplace=True)
log(f"Removed {before - len(df):,} rows with invalid Severity values")
log(f"Severity distribution:\n{df['Severity'].value_counts().to_string()}")

# STEP 5 — FIX COORDINATES
log("\n── STEP 5: FIX COORDINATES")
KARNATAKA_LAT = (11.5, 18.5)
KARNATAKA_LON = (74.0, 78.6)
valid_geo = (
    df['Latitude'].between(*KARNATAKA_LAT) &
    df['Longitude'].between(*KARNATAKA_LON)
)
invalid_count = (~valid_geo).sum()
log(f"Rows with invalid coordinates: {invalid_count:,} ({invalid_count/len(df)*100:.1f}%)")
df.loc[~valid_geo, 'Latitude']  = np.nan
df.loc[~valid_geo, 'Longitude'] = np.nan
valid_after = df['Latitude'].notna().sum()
log(f"Rows with valid coordinates: {valid_after:,} ({valid_after/len(df)*100:.1f}%)")
# Fill missing coordinates with district centroid
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

filled = 0
for idx, row in df[df['Latitude'].isna()].iterrows():
    district = row['DISTRICTNAME']
    if district in DISTRICT_CENTROIDS:
        lat, lon = DISTRICT_CENTROIDS[district]
        # Add small random offset so points don't all stack on same spot
        df.at[idx, 'Latitude']  = lat + np.random.uniform(-0.15, 0.15)
        df.at[idx, 'Longitude'] = lon + np.random.uniform(-0.15, 0.15)
        filled += 1

log(f"Filled {filled:,} missing coordinates using district centroids")
log(f"Remaining nulls: {df['Latitude'].isna().sum():,}")
# STEP 6 — HANDLE OUTLIERS
log("\n── STEP 6: HANDLE OUTLIERS")
df['Noofvehicle_involved'] = df['Noofvehicle_involved'].clip(1, 10)
log(f"Vehicles involved capped at 10")

# STEP 7 — FILL MISSING VALUES
log("\n── STEP 7: FILL MISSING VALUES")
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
log(f"Remaining nulls: {df.isnull().sum().sum()}")

# STEP 8 — ENCODE SEVERITY
log("\n── STEP 8: ENCODE SEVERITY -> RISK LEVEL")
severity_map = {'Fatal': 3, 'Grievous Injury': 2, 'Simple Injury': 1,
                'Damage Only': 0, 'Not Applicable': 0}
df['severity_numeric'] = df['Severity'].map(severity_map)
df['risk_level'] = df['severity_numeric'].apply(
    lambda x: 2 if x == 3 else 1 if x == 2 else 0)
log(f"Risk level distribution:\n{df['risk_level'].value_counts().sort_index().to_string()}")

# STEP 9 — ENCODE CATEGORICALS
log("\n── STEP 9: ENCODE CATEGORICAL FEATURES")
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

surface_map = {'Dry': 2, 'Fine': 2, 'Wet': 1, 'Others': 1,
               'Not Applicable': 1, 'Muddy': 0, 'Ditch or Potholed': 0, 'Flooded': 0}
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
    lambda x: 1 if any(w in str(x).lower() for w in
        ['rain', 'fog', 'mist', 'snow', 'flood', 'wind', 'hail', 'dust', 'cold']) else 0)

df['hit_run'] = (df['Hit_Run'] == 'Yes').astype(int)

cause_map = {'Human Error': 2, 'Vehicle Defect': 1,
             'Road Environment Defect': 1, 'Not Applicable': 0}
df['main_cause_encoded'] = df['Main_Cause'].map(cause_map).fillna(0).astype(int)

df['is_highway'] = df['Accident_Spot'].apply(
    lambda x: 1 if any(w in str(x).lower() for w in ['highway', 'nh', 'sh', 'national']) else 0)

road_char_map = {'Straight Road': 0, 'Curve': 1, 'Curved': 1,
                 'Bridge': 2, 'Culvert': 2, 'Intersection': 2,
                 'Junction': 2, 'Not Applicable': 0, 'Others': 0}
df['road_character_encoded'] = df['Road_Character'].map(road_char_map).fillna(0).astype(int)

df['is_urban'] = df['Accident_Location'].apply(
    lambda x: 1 if any(w in str(x).lower() for w in
        ['city', 'town', 'urban', 'municipal', 'market', 'school', 'hospital']) else 0)

log("Encoded: road_type, road_condition, junction_control, weather_risk,")
log("         hit_run, main_cause, is_highway, road_character, is_urban")

# STEP 10 — ACCIDENT COUNT PROXY
log("\n── STEP 10: ENGINEER ACCIDENT COUNT PROXY")
district_road_counts = df.groupby(['DISTRICTNAME', 'Road_Type']).size().reset_index(name='accident_count_6mo')
df = df.merge(district_road_counts, on=['DISTRICTNAME', 'Road_Type'], how='left')
df['accident_count_6mo'] = df['accident_count_6mo'].clip(0, 500)
log(f"accident_count_6mo range: {df['accident_count_6mo'].min()} - {df['accident_count_6mo'].max()}")

# STEP 11 — HANDLE CLASS IMBALANCE
log("\n── STEP 11: HANDLE CLASS IMBALANCE")
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
df_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
log(f"After balancing:\n{df_balanced['risk_level'].value_counts().sort_index().to_string()}")

# STEP 12 — SELECT FINAL FEATURES
log("\n── STEP 12: SELECT FINAL FEATURES")
ML_FEATURES = [
    'accident_count_6mo', 'severity_numeric', 'road_type_encoded',
    'road_condition', 'junction_control', 'weather_risk',
    'Noofvehicle_involved', 'main_cause_encoded', 'hit_run',
    'is_highway', 'road_character_encoded', 'is_urban',
    'risk_level', 'Latitude', 'Longitude', 'DISTRICTNAME', 'Year',
]
df_ml = df_balanced[ML_FEATURES].copy()
df_ml.rename(columns={'Noofvehicle_involved': 'vehicles_avg'}, inplace=True)
log(f"Final dataset: {df_ml.shape[0]:,} rows x {df_ml.shape[1]} columns")

# STEP 13 — SAVE
log("\n── STEP 13: SAVE")
df_ml.to_csv(OUT_PATH, index=False)
log(f"Saved -> {OUT_PATH}")
df.to_csv(os.path.join(BASE_DIR, 'data', 'cleaned_accidents_full.csv'), index=False)

# SUMMARY
log("\n" + "=" * 60)
log("PREPROCESSING COMPLETE")
log("=" * 60)
log(f"Original : {original_shape[0]:,} rows x {original_shape[1]} cols")
log(f"Cleaned  : {df_ml.shape[0]:,} rows x {df_ml.shape[1]} cols")
log(f"Output   : ml/data/cleaned_accidents.csv")
log(f"Next     : python3 ml/train.py")

with open(REPORT, 'w') as f:
    f.write('\n'.join(lines))
log(f"Report saved -> {REPORT}")
