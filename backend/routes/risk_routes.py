from flask import Blueprint, request, jsonify
import sys, os

# Add ml folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'ml'))

risk_bp = Blueprint('risk', __name__)

_predictor = None


def get_predictor():
    global _predictor

    if _predictor is None:
        try:
            from predict import RiskPredictor

            base_dir = os.path.dirname(__file__)
            ml_dir = os.path.abspath(os.path.join(base_dir, '..', '..', 'ml'))
            model_path = os.path.join(ml_dir, 'outputs', 'best_model.pkl')

            # Only best_model.pkl is required here. feature_columns.pkl and
            # the locality/cluster JSONs are optional — RiskPredictor already
            # resolves those against its own correct default paths (relative
            # to ml/predict.py) and falls back gracefully if they're absent.
            # (There is no scaler.pkl in this pipeline: train.py only fits a
            # StandardScaler inside the Logistic Regression branch, which
            # didn't win, and no model here needs a separately-saved scaler.)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Missing file: {model_path}. Run ml/train.py first.")

            _predictor = RiskPredictor(model_path)

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Model loading failed: {str(e)}")

    return _predictor


# ------------------ SINGLE PREDICT ------------------ #
@risk_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        if not data or 'segment' not in data:
            return jsonify({'error': 'Missing segment data'}), 400

        result = get_predictor().predict(data['segment'])

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ------------------ BATCH PREDICT ------------------ #
@risk_bp.route('/predict/batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json(force=True)

        if not data or 'segments' not in data:
            return jsonify({'error': 'Missing segments list'}), 400

        results = get_predictor().batch_predict(data['segments'])

        return jsonify({
            'success': True,
            'count': len(results),
            'results': results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ------------------ METRICS ------------------ #
@risk_bp.route('/metrics', methods=['GET'])
def model_metrics():
    try:
        metrics_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'ml', 'metrics.json'
        )

        if not os.path.exists(metrics_path):
            return jsonify({'error': 'Model not trained yet'}), 404

        import json
        with open(metrics_path) as f:
            data = json.load(f)

        return jsonify({
            'success': True,
            'metrics': data
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500