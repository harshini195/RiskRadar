"""Risk prediction endpoints."""
from flask import Blueprint, request, jsonify
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml'))

risk_bp = Blueprint('risk', __name__)

_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        from train import RiskPredictor
        from flask import current_app
        _predictor = RiskPredictor(
            current_app.config['MODEL_PATH'],
            current_app.config['SCALER_PATH'],
        )
    return _predictor


@risk_bp.route('/predict', methods=['POST'])
def predict():
    """
    POST /api/risk/predict
    Body: { segment: { accident_count_6mo, severity_avg, road_type_encoded,
                        road_condition, junction_control, weather_risk,
                        vehicles_avg, speed_limit } }
    """
    data = request.get_json(force=True)
    if not data or 'segment' not in data:
        return jsonify({'error': 'Missing segment data'}), 400
    result = get_predictor().predict(data['segment'])
    return jsonify(result)


@risk_bp.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    POST /api/risk/predict/batch
    Body: { segments: [ {...}, ... ] }
    """
    data = request.get_json(force=True)
    if not data or 'segments' not in data:
        return jsonify({'error': 'Missing segments list'}), 400
    results = get_predictor().batch_predict(data['segments'])
    return jsonify({'results': results, 'count': len(results)})


@risk_bp.route('/metrics', methods=['GET'])
def model_metrics():
    """Return stored model evaluation metrics."""
    import json
    metrics_path = os.path.join(
        os.path.dirname(__file__), '..', 'ml', 'metrics.json'
    )
    if not os.path.exists(metrics_path):
        return jsonify({'error': 'Model not trained yet'}), 404
    with open(metrics_path) as f:
        return jsonify(json.load(f))
