"""
Accident Risk Prediction - train.py
Dataset : cleaned_accidents.csv
Target  : risk_level  (0=Low, 1=Medium, 2=High)
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
warnings.filterwarnings("ignore")

DATA_PATH     = r"C:\Users\HP\RiskRadar\ml\data\cleaned_accidents.csv"
TARGET_COLUMN = "risk_level"
FEATURE_COLUMNS = [
    "log_accident_count", "road_type_encoded",
    "road_condition", "junction_control", "weather_risk", "vehicles_avg",
    "main_cause_encoded", "hit_run", "road_character_encoded", "is_urban",
    "year_recency", "urban_road", "risk_junction",
]
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}
OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60); print("STEP 1: Loading data..."); print("="*60)
df = pd.read_csv(DATA_PATH, encoding="latin1")
print(f"  Loaded {len(df):,} rows")

print("\nSTEP 2: Cleaning and engineering features...")
before = len(df)
df = df[df[TARGET_COLUMN].isin([0,1,2])].copy()
df["accident_count_6mo"] = df["accident_count_6mo"].fillna(df["accident_count_6mo"].median())
df["log_accident_count"] = np.log1p(df["accident_count_6mo"])
df["year_recency"]       = df["Year"] - 2016
df["urban_road"]         = df["is_urban"] * df["road_type_encoded"]
df["risk_junction"]      = (df["junction_control"] > 0).astype(int)
df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
print(f"  Remaining: {len(df):,} | Dropped: {before-len(df):,}")
print(f"  Class distribution:\n{df[TARGET_COLUMN].value_counts().sort_index().to_string()}")

print("\nSTEP 3: Preparing features...")
X = df[FEATURE_COLUMNS].values
y = df[TARGET_COLUMN].values
print(f"  Shape: {X.shape}")
with open(os.path.join(OUTPUT_DIR, "feature_columns.pkl"), "wb") as f:
    pickle.dump(FEATURE_COLUMNS, f)
print(f"  Saved feature_columns.pkl")

print("\nSTEP 4: Splitting (70/30)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

print("\nSTEP 5: Training models...")
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_leaf=5,
        max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        min_samples_leaf=10, subsample=0.8, random_state=42),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=42))
    ]),
}

results = {}; best_model = None; best_f1 = 0
for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, target_names=[RISK_LABELS[i] for i in range(3)], output_dict=True)
    results[name] = {"f1_weighted": round(f1,4), "per_class": {
        RISK_LABELS[i]: {"precision": round(report[RISK_LABELS[i]]["precision"],4),
                         "recall": round(report[RISK_LABELS[i]]["recall"],4),
                         "f1": round(report[RISK_LABELS[i]]["f1-score"],4)} for i in range(3)}}
    print(f"    F1 (weighted): {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=[RISK_LABELS[i] for i in range(3)]))
    if f1 > best_f1: best_f1 = f1; best_model = (name, model)

print("\nSTEP 6: Feature importance...")
best_name, best_clf = best_model
print(f"  Best: {best_name} (F1={best_f1:.4f})")
clf_inner = best_clf.named_steps["clf"] if hasattr(best_clf, "named_steps") else best_clf
if hasattr(clf_inner, "feature_importances_"):
    feat_imp = sorted(zip(FEATURE_COLUMNS, clf_inner.feature_importances_), key=lambda x: x[1], reverse=True)
    print("\n  Feature Importances:")
    for feat, imp in feat_imp:
        print(f"    {feat:<30} {imp:.4f}  {'█'*int(imp*40)}")
    results["feature_importance"] = {f: round(float(i),4) for f,i in feat_imp}

print("\nSTEP 7: Hotspot report...")
df_full = pd.read_csv(DATA_PATH, encoding="latin1")
df_full = df_full[df_full[TARGET_COLUMN].isin([0,1,2])]
hotspots = (df_full.groupby("DISTRICTNAME")
    .agg(total_accidents=("risk_level","count"), high_risk_count=("risk_level", lambda x:(x==2).sum()))
    .assign(high_risk_rate=lambda d:(d["high_risk_count"]/d["total_accidents"]*100).round(2))
    .sort_values("high_risk_rate", ascending=False).head(15).reset_index())
hotspots.to_json(os.path.join(OUTPUT_DIR,"hotspots.json"), orient="records", indent=2)
print(hotspots[["DISTRICTNAME","total_accidents","high_risk_count","high_risk_rate"]].head(5).to_string(index=False))

print("\nSTEP 8: Saving...")
with open(os.path.join(OUTPUT_DIR,"metrics.json"),"w") as f: json.dump(results,f,indent=2)
with open(os.path.join(OUTPUT_DIR,"best_model.pkl"),"wb") as f: pickle.dump(best_clf,f)
print(f"  Saved best_model.pkl and metrics.json")
print("\n"+"="*60)
print(f"  DONE — Best: {best_name} | F1: {best_f1:.4f}")
print("="*60)