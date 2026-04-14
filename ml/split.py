import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# ── 1. Load your dataset ──────────────────────────────────────────────
df = pd.read_csv("your_accident_data.csv")  # update path as needed

# ── 2. Sanity check before splitting ─────────────────────────────────
print(f"Total rows: {len(df):,}")
print(f"\nClass distribution (severity_label):")
counts = df['severity_label'].value_counts().sort_index()
for label, count in counts.items():
    print(f"  {label:30s}  {count:>6,}  ({count/len(df)*100:.1f}%)")

# ── 3. Define features and target ────────────────────────────────────
TARGET = 'severity_label'

# Drop columns that are leakage risks or non-features
DROP_COLS = [
    TARGET,
    # Add any ID columns, raw text fields, or post-accident columns here
    # e.g. 'accident_id', 'report_number', 'injury_description'
]

X = df.drop(columns=DROP_COLS)
y = df[TARGET]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape:         {y.shape}")

# ── 4. Stratified split (80/20) ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y          # preserves class proportions in both splits
)

print(f"\nTrain size: {len(X_train):,} rows")
print(f"Test size:  {len(X_test):,}  rows")

# ── 5. Verify stratification worked ──────────────────────────────────
print("\nClass proportions after split:")
print(f"  {'Label':<30} {'Full':>8} {'Train':>8} {'Test':>8}")
print(f"  {'-'*56}")
for label in sorted(y.unique()):
    full_pct  = (y == label).mean() * 100
    train_pct = (y_train == label).mean() * 100
    test_pct  = (y_test == label).mean() * 100
    print(f"  {str(label):<30} {full_pct:>7.1f}% {train_pct:>7.1f}% {test_pct:>7.1f}%")

# ── 6. Optional: save splits to disk (avoids re-splitting later) ──────
X_train.to_parquet("X_train.parquet", index=False)
X_test.to_parquet("X_test.parquet",  index=False)
y_train.to_parquet("y_train.parquet", index=False)  # Series → parquet works fine
y_test.to_parquet("y_test.parquet",   index=False)

print("\nSplits saved to parquet.")