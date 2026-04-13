"""
RiskRadar — EDA for Karnataka Accident Dataset
================================================
Dataset: AccidentReports.csv (Karnataka Police Records)
Run: python3 eda.py
Output: eda_report.txt + charts saved to ml/data/plots/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'AccidentReports.csv')
PLOTS_DIR  = os.path.join(BASE_DIR, 'data', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  RISKRADAR — KARNATAKA ACCIDENT DATASET EDA")
print("=" * 60)

df = pd.read_csv(DATA_PATH, encoding='latin1')
print(f"\n[INFO] Loaded {len(df):,} rows × {len(df.columns)} columns")

# ─────────────────────────────────────────────────────────────────────────────
# 1. BASIC INFO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("1. BASIC INFO")
print("─" * 60)
print(f"Total accidents : {len(df):,}")
print(f"Years covered   : {df['Year'].min()} – {df['Year'].max()}")
print(f"Districts       : {df['DISTRICTNAME'].nunique()} unique")
print(f"Police units    : {df['UNITNAME'].nunique()} unique")
print(f"\nColumns ({len(df.columns)}):")
for col in df.columns:
    null_pct = df[col].isnull().mean() * 100
    print(f"  {col:<35} dtype={df[col].dtype}  nulls={null_pct:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 2. MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("2. MISSING VALUES")
print("─" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
if len(missing_df) > 0:
    print(missing_df.to_string())
else:
    print("No missing values!")

# ─────────────────────────────────────────────────────────────────────────────
# 3. SEVERITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("3. SEVERITY ANALYSIS")
print("─" * 60)
sev_counts = df['Severity'].value_counts()
print(sev_counts.to_string())
print(f"\nMost common severity: {sev_counts.index[0]} ({sev_counts.iloc[0]:,} cases)")

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#ef4444' if 'fatal' in str(s).lower() or 'death' in str(s).lower()
          else '#f59e0b' if 'grievous' in str(s).lower() or 'serious' in str(s).lower()
          else '#22c55e' for s in sev_counts.index]
bars = ax.barh(sev_counts.index, sev_counts.values, color=colors)
ax.set_xlabel('Number of Accidents')
ax.set_title('Accident Severity Distribution — Karnataka', fontsize=13, fontweight='bold')
for bar, val in zip(bars, sev_counts.values):
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '01_severity.png'), dpi=150)
plt.close()
print("[SAVED] 01_severity.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. YEAR-WISE TREND
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("4. YEAR-WISE ACCIDENT TREND")
print("─" * 60)
year_counts = df['Year'].value_counts().sort_index()
print(year_counts.to_string())

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(year_counts.index, year_counts.values, marker='o', color='#3b82f6', linewidth=2.5, markersize=7)
ax.fill_between(year_counts.index, year_counts.values, alpha=0.15, color='#3b82f6')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Accidents')
ax.set_title('Year-wise Accident Trend — Karnataka', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for x, y in zip(year_counts.index, year_counts.values):
    ax.annotate(f'{y:,}', (x, y), textcoords='offset points', xytext=(0, 8),
                ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '02_yearly_trend.png'), dpi=150)
plt.close()
print("[SAVED] 02_yearly_trend.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. TOP DISTRICTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("5. TOP 15 ACCIDENT-PRONE DISTRICTS")
print("─" * 60)
district_counts = df['DISTRICTNAME'].value_counts().head(15)
print(district_counts.to_string())

fig, ax = plt.subplots(figsize=(9, 7))
colors_d = ['#ef4444' if i < 3 else '#f59e0b' if i < 7 else '#3b82f6'
            for i in range(len(district_counts))]
bars = ax.barh(district_counts.index[::-1], district_counts.values[::-1], color=colors_d[::-1])
ax.set_xlabel('Number of Accidents')
ax.set_title('Top 15 Accident-Prone Districts — Karnataka', fontsize=13, fontweight='bold')
for bar, val in zip(bars, district_counts.values[::-1]):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '03_districts.png'), dpi=150)
plt.close()
print("[SAVED] 03_districts.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN CAUSE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("6. MAIN CAUSES OF ACCIDENTS")
print("─" * 60)
cause_counts = df['Main_Cause'].value_counts().head(12)
print(cause_counts.to_string())

fig, ax = plt.subplots(figsize=(9, 6))
wedges, texts, autotexts = ax.pie(
    cause_counts.values[:8],
    labels=cause_counts.index[:8],
    autopct='%1.1f%%',
    startangle=140,
    colors=plt.cm.Set3.colors
)
ax.set_title('Top 8 Causes of Accidents — Karnataka', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '04_causes.png'), dpi=150)
plt.close()
print("[SAVED] 04_causes.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. ROAD TYPE & CONDITION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("7. ROAD TYPE DISTRIBUTION")
print("─" * 60)
road_counts = df['Road_Type'].value_counts()
print(road_counts.to_string())

print("\nROAD CONDITION:")
rc_counts = df['Road_Condition'].value_counts()
print(rc_counts.to_string())

print("\nSURFACE CONDITION:")
sc_counts = df['Surface_Condition'].value_counts()
print(sc_counts.to_string())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (data, title) in zip(axes, [
    (road_counts, 'Road Type'),
    (rc_counts,   'Road Condition'),
    (sc_counts,   'Surface Condition'),
]):
    top = data.head(6)
    ax.bar(range(len(top)), top.values, color='#3b82f6', alpha=0.8)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top.index, rotation=30, ha='right', fontsize=8)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Count')
    ax.grid(axis='y', alpha=0.3)
plt.suptitle('Road Characteristics — Karnataka Accidents', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '05_road_characteristics.png'), dpi=150)
plt.close()
print("[SAVED] 05_road_characteristics.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. WEATHER & JUNCTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("8. WEATHER CONDITIONS")
print("─" * 60)
weather_counts = df['Weather'].value_counts()
print(weather_counts.to_string())

print("\nJUNCTION CONTROL:")
jn_counts = df['Junction_Control'].value_counts()
print(jn_counts.to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(weather_counts.index[:6], weather_counts.values[:6], color='#0d9488', alpha=0.85)
axes[0].set_title('Weather Conditions', fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=30)
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(jn_counts.index[:6], jn_counts.values[:6], color='#7c3aed', alpha=0.85)
axes[1].set_title('Junction Control', fontweight='bold')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=30)
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('Environmental Factors — Karnataka Accidents', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '06_weather_junction.png'), dpi=150)
plt.close()
print("[SAVED] 06_weather_junction.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. VEHICLES INVOLVED
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("9. VEHICLES INVOLVED")
print("─" * 60)
veh = df['Noofvehicle_involved'].dropna()
print(f"Min: {veh.min():.0f}  Max: {veh.max():.0f}  Mean: {veh.mean():.2f}  Median: {veh.median():.0f}")
print(veh.value_counts().head(8).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 10. COLLISION TYPE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("10. COLLISION TYPE")
print("─" * 60)
col_counts = df['Collision_Type'].value_counts()
print(col_counts.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 11. GEOSPATIAL OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("11. GEOSPATIAL OVERVIEW")
print("─" * 60)
geo = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)].dropna(subset=['Latitude', 'Longitude'])
print(f"Rows with valid coordinates: {len(geo):,} / {len(df):,}")
print(f"Lat range: {geo['Latitude'].min():.4f} – {geo['Latitude'].max():.4f}")
print(f"Lon range: {geo['Longitude'].min():.4f} – {geo['Longitude'].max():.4f}")

if len(geo) > 100:
    fig, ax = plt.subplots(figsize=(9, 9))
    severity_color = geo['Severity'].apply(
        lambda s: '#ef4444' if 'fatal' in str(s).lower() or 'death' in str(s).lower()
        else '#f59e0b' if 'grievous' in str(s).lower()
        else '#22c55e'
    )
    ax.scatter(geo['Longitude'], geo['Latitude'],
               c=severity_color, alpha=0.3, s=5)
    red_patch   = mpatches.Patch(color='#ef4444', label='Fatal')
    amber_patch = mpatches.Patch(color='#f59e0b', label='Grievous')
    green_patch = mpatches.Patch(color='#22c55e', label='Minor')
    ax.legend(handles=[red_patch, amber_patch, green_patch], loc='upper left')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Accident Locations — Karnataka', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '07_geo_scatter.png'), dpi=150)
    plt.close()
    print("[SAVED] 07_geo_scatter.png")

# ─────────────────────────────────────────────────────────────────────────────
# 12. SEVERITY vs ROAD TYPE CROSSTAB
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("12. SEVERITY vs ROAD TYPE")
print("─" * 60)
cross = pd.crosstab(df['Road_Type'], df['Severity'])
print(cross.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 13. HIT & RUN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("13. HIT & RUN CASES")
print("─" * 60)
hr = df['Hit_Run'].value_counts()
print(hr.to_string())
hr_pct = (hr.get('Yes', 0) / len(df) * 100)
print(f"\nHit & Run rate: {hr_pct:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 14. KEY INSIGHTS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY INSIGHTS FOR RISKRADAR")
print("=" * 60)
print(f"""
Dataset Summary:
  - Total accidents     : {len(df):,}
  - Districts covered   : {df['DISTRICTNAME'].nunique()}
  - Year range          : {df['Year'].min()} – {df['Year'].max()}
  - Valid coordinates   : {len(geo):,} ({len(geo)/len(df)*100:.1f}%)

Top Risk Factors:
  - Most dangerous district  : {district_counts.index[0]}
  - Most common cause        : {cause_counts.index[0]}
  - Most common road type    : {road_counts.index[0]}
  - Most common weather      : {weather_counts.index[0]}
  - Hit & Run rate           : {hr_pct:.1f}%

ML Features available in this dataset:
  ✓ Severity           → risk label (target variable)
  ✓ Road_Type          → road_type_encoded
  ✓ Road_Condition     → road_condition
  ✓ Junction_Control   → junction_control
  ✓ Weather            → weather_risk
  ✓ Noofvehicle_involved → vehicles_avg
  ✓ Main_Cause         → cause encoding
  ✓ Latitude/Longitude → hotspot clustering (DBSCAN)

Plots saved to: {PLOTS_DIR}
""")
