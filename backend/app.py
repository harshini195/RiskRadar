from flask import Flask, request, make_response, jsonify
import sys, os
from flask_cors import CORS
from backend.routes.risk_routes import risk_bp
from backend.routes.route_routes import route_bp
from backend.routes.hotspot_routes import hotspot_bp
from backend.config import Config
from werkzeug.exceptions import HTTPException
from ml.predict import RiskPredictor
import json
import logging
import traceback
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

    # Ensure CORS headers are present on all responses (including errors) and
    # provide basic handling for OPTIONS preflight requests.
    @app.after_request
    def add_cors_headers(response):
        response.headers.setdefault('Access-Control-Allow-Origin', '*')
        response.headers.setdefault('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.setdefault('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
        return response

    @app.route('/api/<path:path>', methods=['OPTIONS'])
    def catch_all_options(path):
        resp = make_response('', 204)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        resp.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'
        return resp

    app.register_blueprint(risk_bp, url_prefix='/api/risk')
    app.register_blueprint(route_bp, url_prefix='/api/routes')
    app.register_blueprint(hotspot_bp, url_prefix='/api/hotspots')

    @app.route('/api/health')
    def health():
        return {'status': 'ok', 'service': 'RiskRadar API'}

    @app.route('/api/risk/model-metrics')
    def model_metrics():
        import os
        import json

        metrics_path = os.path.join(os.path.dirname(__file__), '../ml/outputs/metrics.json')


        try:
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)

        # Use Random Forest metrics
            rf = metrics_data.get("Random Forest", {})
            classes = rf.get("per_class", {})

            if not classes:
                return {"error": "No class metrics found"}, 500

            precision = sum(c["precision"] for c in classes.values()) / len(classes)
            recall = sum(c["recall"] for c in classes.values()) / len(classes)
            f1 = rf.get("f1_weighted", 0)

            # Approximate accuracy
            accuracy = recall

            metrics = [
                {"label": "Accuracy",  "value": f"{accuracy*100:.1f}%", "color": "#22c55e", "pct": accuracy*100},
                {"label": "Precision", "value": f"{precision*100:.1f}%", "color": "#4f8ef7", "pct": precision*100},
                {"label": "Recall",    "value": f"{recall*100:.1f}%", "color": "#f59e0b", "pct": recall*100},
                {"label": "F1 Score",  "value": f"{f1:.3f}", "color": "#4f8ef7", "pct": f1*100},
            ]

            # Feature importance
            features = []
            fi = metrics_data.get("feature_importance", {})

            if fi:
                total = sum(abs(v) for v in fi.values()) or 1

                for name, val in fi.items():
                    pct = round(abs(val) / total * 100)
                    features.append({
                        "name": name.replace('_', ' ').title(),
                        "pct": pct
                    })

            features = sorted(features, key=lambda x: -x['pct'])

            return {"metrics": metrics, "features": features}
        except Exception as e:
            print("ERROR:", e)   
            return {"error": str(e)}, 500
    risk_predictor = RiskPredictor()

    @app.route('/predict', methods=['POST'])
    def predict_risk():
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No input data provided'}), 400
            # Accept single segment or list of segments
            segments = data if isinstance(data, list) else [data]
            results = []
            for seg in segments:
                engineered = RiskPredictor.engineer(seg)
                result = risk_predictor.predict(engineered)
                results.append(result)
            # If only one segment, return as object
            if isinstance(data, dict):
                return jsonify(results[0])
            return jsonify(results)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.errorhandler(HTTPException)
    def handle_http_exception(e):
        response = e.get_response()
        response.data = json.dumps({
            "code": e.code,
            "name": e.name,
            "description": e.description,
        })
        response.content_type = "application/json"
        response.headers.setdefault('Access-Control-Allow-Origin', '*')
        response.headers.setdefault('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.setdefault('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
        return response

    @app.errorhandler(Exception)
    def handle_exception(e):
        logging.error('Unhandled Exception: %s', str(e))
        traceback.print_exc()
        response = make_response(json.dumps({
            "code": 500,
            "name": "Internal Server Error",
            "description": str(e),
        }), 500)
        response.headers.setdefault('Access-Control-Allow-Origin', '*')
        response.headers.setdefault('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.setdefault('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
        response.content_type = "application/json"
        return response

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)