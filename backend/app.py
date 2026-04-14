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
        # Mock data, replace with real model results as needed
        metrics = [
            {"label": "Accuracy",  "value": "87.4%", "color": "#22c55e", "pct": 87.4},
            {"label": "Precision", "value": "83.1%", "color": "#4f8ef7", "pct": 83.1},
            {"label": "Recall",    "value": "91.2%", "color": "#f59e0b", "pct": 91.2},
            {"label": "F1 Score",  "value": "0.871",  "color": "#4f8ef7", "pct": 87.1},
        ]
        features = [
            {"name": "Accident History", "pct": 31},
            {"name": "Road Condition",   "pct": 22},
            {"name": "Junction Type",    "pct": 18},
            {"name": "Traffic Density",  "pct": 14},
            {"name": "Weather Pattern",  "pct": 9},
            {"name": "Speed Limit",      "pct": 6},
        ]
        return {"metrics": metrics, "features": features}

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