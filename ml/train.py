"""
RiskRadar ML Pipeline
Trains a Random Forest model to predict accident risk scores for road segments.

Features:
  - accident_count_6mo    : historical accidents in last 6 months
  - severity_avg          : average severity score (1–3)
  - road_type_encoded     : 0=residential, 1=arterial, 2=highway
  - road_condition        : 0=poor, 1=average, 2=good
  - junction_control      : 0=none, 1=sign, 2=signal, 3=roundabout
  - weather_risk          : 0=clear, 1=rain/fog
  - vehicles_avg          : avg vehicles involved per accident
  - speed_limit           : posted speed limit (km/h)

Target: risk_level  0=low, 1=medium, 2=high
"""

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')


# ─── Synthetic dataset generator ─────────────────────────────────────────────

def generate_accident_data(n=1000, seed=42):
    """Generate synthetic accident records for training."""
    rng = np.random.default_rng(seed)

    road_type      = rng.integers(0, 3, n)
    road_condition = rng.integers(0, 3, n)
    junction       = rng.integers(0, 4, n)
    weather_risk   = rng.integers(0, 2, n)
    speed_limit    = rng.choice([30, 40, 50, 60, 80, 100], n)
    vehicles_avg   = rng.integers(1, 5, n)

    # Accident count influenced by features
    base = (
        road_type * 3
        + (2 - road_condition) * 4
        + (3 - junction) * 2
        + weather_risk * 5
        + (speed_limit / 20)
        + vehicles_avg
    )
    noise = rng.normal(0, 3, n)
    accident_count = np.clip(base + noise, 0, 60).astype(int)
    severity_avg   = np.clip(rng.normal(1.5 + road_type * 0.3, 0.4, n), 1, 3)

    # Risk label from accident count
    risk_level = np.where(accident_count >= 20, 2, np.where(accident_count >= 8, 1, 0))

    df = pd.DataFrame({
        'accident_count_6mo': accident_count,
        'severity_avg':       severity_avg.round(2),
        'road_type_encoded':  road_type,
        'road_condition':     road_condition,
        'junction_control':   junction,
        'weather_risk':       weather_risk,
        'vehicles_avg':       vehicles_avg,
        'speed_limit':        speed_limit,
        'risk_level':         risk_level,
    })
    # Add lat/lon for clustering demo (Bangalore bounding box)
    df['latitude']  = rng.uniform(12.85, 13.15, n)
    df['longitude'] = rng.uniform(77.45, 77.75, n)
    return df


# ─── Hotspot detection via DBSCAN ────────────────────────────────────────────

def detect_hotspots(df, eps_km=0.5, min_samples=5):
    """
    Run DBSCAN on accident coordinates to find hotspot clusters.
    eps_km: neighbourhood radius in kilometres (approx degrees ÷ 111).
    Returns df with 'cluster' column; -1 = noise.
    """
    coords = df[['latitude', 'longitude']].values
    eps_deg = eps_km / 111.0
    labels = DBSCAN(eps=eps_deg, min_samples=min_samples).fit_predict(coords)
    df = df.copy()
    df['cluster'] = labels

    hotspots = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        mask = labels == cid
        cluster_df = df[mask]
        hotspots.append({
            'cluster_id':    int(cid),
            'latitude':      float(cluster_df['latitude'].mean()),
            'longitude':     float(cluster_df['longitude'].mean()),
            'accident_count': int(mask.sum()),
            'avg_severity':  float(cluster_df['severity_avg'].mean().round(2)),
            'risk_class':    int(cluster_df['risk_level'].mode()[0]),
        })
    return df, hotspots


# ─── Model training ──────────────────────────────────────────────────────────

FEATURES = [
    'accident_count_6mo', 'severity_avg', 'road_type_encoded',
    'road_condition', 'junction_control', 'weather_risk',
    'vehicles_avg', 'speed_limit',
]

def train_model(df, save_dir='.'):
    X = df[FEATURES]
    y = df['risk_level']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Compare three models ──────────────────────────────────────────────
    candidates = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            C=1.0, max_iter=1000, class_weight='balanced', random_state=42
        ),
    }

    results = {}
    for name, clf in candidates.items():
        clf.fit(X_train_s, y_train)
        cv = cross_val_score(clf, X_train_s, y_train, cv=5, scoring='f1_weighted')
        results[name] = {
            'model':    clf,
            'cv_mean':  cv.mean(),
            'cv_std':   cv.std(),
            'test_acc': accuracy_score(y_test, clf.predict(X_test_s)),
        }
        print(f"{name}: CV F1={cv.mean():.3f}±{cv.std():.3f}  Test Acc={results[name]['test_acc']:.3f}")

    best_name = max(results, key=lambda k: results[k]['cv_mean'])
    best = results[best_name]
    model = best['model']
    print(f"\nBest model: {best_name}")

    y_pred = model.predict(X_test_s)
    metrics = {
        'model':     best_name,
        'accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred, average='weighted'), 4),
        'recall':    round(recall_score(y_test, y_pred, average='weighted'), 4),
        'f1':        round(f1_score(y_test, y_pred, average='weighted'), 4),
        'report':    classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }
    print(classification_report(y_test, y_pred, target_names=['Low','Medium','High']))

    # Feature importances (RF/GB only)
    if hasattr(model, 'feature_importances_'):
        fi = dict(zip(FEATURES, model.feature_importances_.round(4)))
        metrics['feature_importance'] = fi
        print("Feature importances:", fi)

    # Persist
    import os
    with open(os.path.join(save_dir, 'risk_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model, scaler, and metrics to {save_dir}/")
    return model, scaler, metrics


# ─── Risk scorer (used by Flask API) ─────────────────────────────────────────

class RiskPredictor:
    """Load trained model and score a single road segment."""

    def __init__(self, model_path, scaler_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

    def predict(self, segment: dict) -> dict:
        """
        segment keys: accident_count_6mo, severity_avg, road_type_encoded,
                      road_condition, junction_control, weather_risk,
                      vehicles_avg, speed_limit
        Returns: risk_class (0/1/2), risk_score (0–1), risk_label
        """
        row = [[segment.get(f, 0) for f in FEATURES]]
        row_s = self.scaler.transform(row)

        cls = int(self.model.predict(row_s)[0])
        proba = self.model.predict_proba(row_s)[0]

        # Continuous risk score: weighted sum of class probabilities
        score = float(proba[1] * 0.5 + proba[2] * 1.0)
        score = round(min(score, 1.0), 3)

        labels = {0: 'Low', 1: 'Moderate', 2: 'High'}
        return {
            'risk_class': cls,
            'risk_score': score,
            'risk_label': labels[cls],
            'probabilities': {
                'low':      round(float(proba[0]), 3),
                'moderate': round(float(proba[1]), 3),
                'high':     round(float(proba[2]), 3),
            }
        }

    def batch_predict(self, segments: list) -> list:
        return [self.predict(s) for s in segments]


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== RiskRadar ML Pipeline ===\n")
    df = generate_accident_data(n=1200)
    print(f"Generated {len(df)} accident records")
    print(df[['accident_count_6mo', 'severity_avg', 'risk_level']].describe().round(2))

    df, hotspots = detect_hotspots(df)
    print(f"\nDetected {len(hotspots)} accident hotspot clusters")
    for h in hotspots[:5]:
        print(f"  Cluster {h['cluster_id']}: {h['accident_count']} accidents @ "
              f"({h['latitude']:.4f}, {h['longitude']:.4f})")

    print("\n=== Training Models ===")
    model, scaler, metrics = train_model(df, save_dir='.')
    print(f"\nFinal Metrics: Accuracy={metrics['accuracy']} F1={metrics['f1']}")
