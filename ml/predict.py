"""
predict.py — Loads the saved model for inference.
Features include engineered columns that must be computed at prediction time.
"""
import os
import pickle
import numpy as np
import pandas as pd

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}

DEFAULT_MODEL_PATH   = os.path.join(os.path.dirname(__file__), "outputs", "best_model.pkl")
DEFAULT_FEATURE_PATH = os.path.join(os.path.dirname(__file__), "outputs", "feature_columns.pkl")

# Fallback if feature_columns.pkl is missing
FEATURE_COLUMNS = [
    "log_accident_count",
    "road_type_encoded",
    "road_condition",
    "junction_control",
    "weather_risk",
    "vehicles_avg",
    "main_cause_encoded",
    "hit_run",
    "road_character_encoded",
    "is_urban",
    "year_recency",
    "urban_road",
    "risk_junction",
]


class RiskPredictor:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, feature_path=DEFAULT_FEATURE_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        if os.path.exists(feature_path):
            with open(feature_path, "rb") as f:
                self.feature_columns = pickle.load(f)
        else:
            self.feature_columns = FEATURE_COLUMNS

        print(f"[RiskPredictor] Loaded from {model_path}")

    @staticmethod
    def engineer(segment: dict) -> dict:
        """
        Compute derived features from a raw segment dict.
        Call this before passing to predict() if your segment uses
        raw fields (accident_count_6mo, Year) instead of engineered ones.

        Raw inputs expected:
            accident_count_6mo, Year, is_urban, road_type_encoded, junction_control
        """
        import math
        s = dict(segment)
        acc = float(s.get("accident_count_6mo", 8))
        year = int(s.get("Year", 2023))
        s["log_accident_count"] = math.log1p(acc)
        s["year_recency"]       = year - 2016
        s["urban_road"]         = int(s.get("is_urban", 0)) * int(s.get("road_type_encoded", 0))
        s["risk_junction"]      = 1 if int(s.get("junction_control", 0)) > 0 else 0
        return s

    def _build_feature_vector(self, segment: dict) -> pd.DataFrame:
        row = []
        for col in self.feature_columns:
            val = segment.get(col, 0)
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = 0.0
            row.append(val)
        return pd.DataFrame([row], columns=self.feature_columns)

    def predict(self, segment: dict) -> dict:
        """
        Predict risk level for a single road segment.

        segment must contain engineered features. Use RiskPredictor.engineer()
        to compute them from raw fields if needed.

        Required keys:
            severity_numeric, log_accident_count, road_type_encoded,
            road_condition, junction_control, weather_risk, vehicles_avg,
            main_cause_encoded, hit_run, road_character_encoded, is_urban,
            year_recency, urban_road, risk_junction

        Returns:
            { risk_level: int, risk_label: str, probabilities: dict }
        """
        X    = self._build_feature_vector(segment)
        pred = int(self.model.predict(X)[0])

        result = {
            "risk_level": pred,
            "risk_label": RISK_LABELS.get(pred, "Unknown"),
        }

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0].tolist()
            result["probabilities"] = {
                RISK_LABELS[i]: round(probs[i], 4)
                for i in range(len(probs)) if i in RISK_LABELS
            }

        return result

    def batch_predict(self, segments: list) -> list:
        return [self.predict(s) for s in segments]


# Smoke-test
if __name__ == "__main__":
    predictor = RiskPredictor()

    raw = {
        "accident_count_6mo":    12,
        "road_type_encoded":     3,
        "road_condition":        1,
        "junction_control":      1,
        "weather_risk":          0,
        "vehicles_avg":          2,
        "main_cause_encoded":    2,
        "hit_run":               0,
        "road_character_encoded": 0,
        "is_urban":              1,
        "Year":                  2023,
    }

    segment = RiskPredictor.engineer(raw)
    result  = predictor.predict(segment)
    print(f"\nRisk Level : {result['risk_level']} ({result['risk_label']})")
    if "probabilities" in result:
        for label, prob in result["probabilities"].items():
            print(f"  {label:8s}: {prob:.4f}")