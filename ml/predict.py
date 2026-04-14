"""
predict.py — Loads the saved model and encoders for inference.
Never imports train.py or reads the CSV.
"""

import os
import pickle
import numpy as np

# These must match what train.py used
FEATURE_COLUMNS = [
    "Main_Cause",
    "Weather",
    "Road_Type",
    "Collision_Type",
    "Road_Character",
    "Surface_Condition",
    "Road_Condition",
    "Junction_Control",
    "Noofvehicle_involved",
    "Hit_Run",
    "Lane_Type",
    "Year",
    "DISTRICTNAME",
]

SEVERITY_ORDER = ["Damage Only", "Simple Injury", "Grievous Injury", "Fatal"]

# Default path — can be overridden
DEFAULT_MODEL_PATH   = os.path.join(os.path.dirname(__file__), "outputs", "best_model.pkl")
DEFAULT_ENCODER_PATH = os.path.join(os.path.dirname(__file__), "outputs", "label_encoders.pkl")


class RiskPredictor:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, encoder_path=DEFAULT_ENCODER_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoders not found: {encoder_path}. Run train.py first.")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(encoder_path, "rb") as f:
            enc = pickle.load(f)
            self.feature_encoders = enc["features"]
            # target encoder may be OrdinalEncoder or LabelEncoder
            self.target_encoder = enc.get("target")

        print(f"[RiskPredictor] Model loaded from {model_path}")

    def _encode_segment(self, segment: dict) -> np.ndarray:
        """
        Encode a single segment dict into a feature vector.
        Unknown values fall back to 0 (first encoded category).
        """
        row = []
        for col in FEATURE_COLUMNS:
            val = segment.get(col, "Unknown")
            if col in self.feature_encoders:
                le = self.feature_encoders[col]
                # If unseen value, default to 0
                if str(val) in le.classes_:
                    val = le.transform([str(val)])[0]
                else:
                    val = 0
            else:
                # Numeric column
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    val = 0.0
            row.append(val)
        return np.array(row).reshape(1, -1)

    def predict(self, segment: dict) -> dict:
        """
        Predict severity for a single segment.
        Returns: { severity, severity_index, probabilities }
        """
        X = self._encode_segment(segment)
        pred_index = int(self.model.predict(X)[0])
        severity = SEVERITY_ORDER[pred_index] if pred_index < len(SEVERITY_ORDER) else "Unknown"

        result = {
            "severity": severity,
            "severity_index": pred_index,  # 0=Damage Only ... 3=Fatal
        }

        # Add probabilities if model supports it
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0].tolist()
            result["probabilities"] = {
                SEVERITY_ORDER[i]: round(probs[i], 4)
                for i in range(len(probs))
                if i < len(SEVERITY_ORDER)
            }

        return result

    def batch_predict(self, segments: list) -> list:
        """Predict severity for a list of segments."""
        return [self.predict(s) for s in segments]