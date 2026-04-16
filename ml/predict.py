"""
predict.py — Loads the saved model for inference.
Features include engineered columns that must be computed at prediction time.
Locality is treated as a FIRST-CLASS feature with pre-computed encodings.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}
RISK_SCORE_MAP = {0: 0.2, 1: 0.55, 2: 0.85}

DEFAULT_MODEL_PATH   = os.path.join(os.path.dirname(__file__), "outputs", "best_model.pkl")
DEFAULT_FEATURE_PATH = os.path.join(os.path.dirname(__file__), "outputs", "feature_columns.pkl")
LOCALITY_ENC_PATH    = os.path.join(os.path.dirname(__file__), "outputs", "locality_encodings.json")
CLUSTER_MAP_PATH     = os.path.join(os.path.dirname(__file__), "outputs", "locality_cluster_map.json")

# Fallback if feature_columns.pkl is missing
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

        # Load locality encodings
        self.locality_encodings = {}
        if os.path.exists(LOCALITY_ENC_PATH):
            with open(LOCALITY_ENC_PATH, "r") as f:
                enc_list = json.load(f)
                for item in enc_list:
                    loc = item["locality"]
                    self.locality_encodings[loc] = {
                        "locality_freq_enc": item["locality_freq_enc"],
                        "locality_risk_enc": item["locality_risk_enc"],
                        "locality_high_risk_flag": item["locality_high_risk_flag"],
                        "locality_sev_mean": item["locality_sev_mean"],
                    }
        
        # Load geo-cluster map (locality -> cluster ID)
        self.cluster_map = {}
        if os.path.exists(CLUSTER_MAP_PATH):
            with open(CLUSTER_MAP_PATH, "r") as f:
                self.cluster_map = json.load(f)

        # Default values for unknown localities
        self.default_locality_enc = {
            "locality_freq_enc": 1,
            "locality_risk_enc": 1.0,  # neutral risk
            "locality_high_risk_flag": 0,
            "locality_sev_mean": 2.0,  # median severity
        }
        self.default_geo_cluster = -1  # noise cluster

        print(f"[RiskPredictor] Loaded from {model_path}")
        print(f"  Features       : {len(self.feature_columns)}")
        print(f"  Locality info  : {len(self.locality_encodings)} localities")
        print(f"  Geo clusters   : {len(set(self.cluster_map.values()))} unique clusters")

    @staticmethod
    def engineer(segment: dict) -> dict:
        """
        Engineer features from raw segment data.
        
        Required raw inputs:
            - accident_count_6mo (or default to 8)
            - Noofvehicle_involved (or default to 2)
            - Year (or default to 2023)
            - is_urban (0 or 1)
            - road_type_encoded
            - junction_control
            - weather_risk
            - road_condition
            - locality (string, uppercase)
            - locality_accident_count
            - locality_log_volume
            
        Returns:
            Dictionary with all engineered features
        """
        s = dict(segment)
        
        # Parse numeric values
        acc = float(s.get("accident_count_6mo", 8))
        vehicles = float(s.get("Noofvehicle_involved", 2))
        year = int(s.get("Year", 2023))
        is_urban = int(s.get("is_urban", 0))
        road_type = int(s.get("road_type_encoded", 0))
        junction = int(s.get("junction_control", 0))
        weather = int(s.get("weather_risk", 0))
        road_cond = int(s.get("road_condition", 0))
        
        # Basic transformations
        s["vehicles_avg"] = vehicles
        s["log_accident_count"] = np.log1p(acc)
        s["accident_sqrt"] = np.sqrt(acc)
        s["vehicles_log"] = np.log1p(vehicles)
        s["year_recency"] = year - 2016
        s["risk_junction"] = 1 if junction > 0 else 0
        
        # Interaction features
        s["urban_road"] = is_urban * road_type
        s["urban_traffic"] = is_urban * vehicles
        s["weather_road"] = weather * road_cond
        s["junction_traffic"] = junction * vehicles
        s["risk_weather_vehicle"] = weather * vehicles
        s["urban_junction"] = is_urban * junction
        s["traffic_intensity"] = vehicles * road_type
        s["busy_junction"] = vehicles * junction
        
        return s

    def add_locality_features(self, segment: dict) -> dict:
        """
        Add locality-specific features using pre-computed encodings.
        
        Expects segment["locality"] to be present (uppercase string).
        """
        s = dict(segment)
        
        # Normalize locality name
        locality = str(s.get("locality", "UNKNOWN")).strip().upper()
        
        # Get locality encodings (or use defaults)
        loc_enc = self.locality_encodings.get(locality, self.default_locality_enc)
        s["locality_freq_enc"] = loc_enc["locality_freq_enc"]
        s["locality_risk_enc"] = loc_enc["locality_risk_enc"]
        s["locality_high_risk_flag"] = loc_enc["locality_high_risk_flag"]
        s["locality_sev_mean"] = loc_enc["locality_sev_mean"]
        
        # Get geo cluster (or use default)
        s["geo_cluster"] = self.cluster_map.get(locality, self.default_geo_cluster)
        
        # Locality interaction features (if needed by model)
        if "locality_urban_risk" in self.feature_columns:
            s["locality_urban_risk"] = s["locality_risk_enc"] * s.get("is_urban", 0)
        if "locality_weather_risk" in self.feature_columns:
            s["locality_weather_risk"] = s["locality_risk_enc"] * s.get("weather_risk", 0)
        if "locality_sev_volume" in self.feature_columns:
            s["locality_sev_volume"] = s["locality_sev_mean"] * s.get("locality_log_volume", 0)
        
        return s

    def _build_feature_vector(self, segment: dict) -> pd.DataFrame:
        """Build feature vector in the exact order expected by the model."""
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

        segment can be either:
          1. Raw data (will be engineered automatically)
          2. Pre-engineered data (must contain all required features)

        Returns:
            { risk_level: int, risk_label: str, risk_score: float, probabilities: dict }
        """
        # Auto-engineer if raw data detected
        if "log_accident_count" not in segment:
            segment = self.engineer(segment)
        
        # Add locality features
        segment = self.add_locality_features(segment)
        
        # Build feature vector
        X = self._build_feature_vector(segment)
        
        # Predict
        pred = int(self.model.predict(X)[0])

        result = {
            "risk_level": pred,
            "risk_label": RISK_LABELS.get(pred, "Unknown"),
            "risk_score": RISK_SCORE_MAP.get(pred, 0.2),
        }

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0].tolist()
            result["probabilities"] = {
                RISK_LABELS[i]: round(probs[i], 4)
                for i in range(len(probs)) if i in RISK_LABELS
            }

        return result

    def batch_predict(self, segments: list) -> list:
        """Predict risk for multiple segments."""
        return [self.predict(s) for s in segments]


# Smoke-test
if __name__ == "__main__":
    predictor = RiskPredictor()

    # Test with raw data
    raw = {
        "locality": "MG ROAD",
        "accident_count_6mo": 12,
        "Noofvehicle_involved": 2,
        "road_type_encoded": 3,
        "road_condition": 1,
        "junction_control": 1,
        "weather_risk": 0,
        "main_cause_encoded": 2,
        "hit_run": 0,
        "road_character_encoded": 0,
        "is_urban": 1,
        "is_highway": 0,
        "Year": 2023,
        "locality_accident_count": 450,
        "locality_fatal_count": 25,
        "locality_high_sev_count": 180,
        "locality_fatal_rate": 5.6,
        "locality_high_sev_rate": 40.0,
        "locality_log_volume": 6.1,
        "locality_risk_score": 1.85,
        "locality_risk_rank": 12,
    }

    print("\n" + "=" * 60)
    print("PREDICTION TEST")
    print("=" * 60)
    result = predictor.predict(raw)
    print(f"\nRisk Level : {result['risk_level']} ({result['risk_label']})")
    print(f"Risk Score : {result['risk_score']}")
    if "probabilities" in result:
        print("\nProbabilities:")
        for label, prob in result["probabilities"].items():
            bar = "█" * int(prob * 40)
            print(f"  {label:8s}: {prob:.4f}  {bar}")
    print("=" * 60)