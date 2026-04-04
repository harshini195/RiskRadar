from flask import Flask, request, make_response
from flask_cors import CORS
from routes.risk_routes import risk_bp
from routes.route_routes import route_bp
from routes.hotspot_routes import hotspot_bp
from config import Config
from werkzeug.exceptions import HTTPException
import json
import logging
import traceback

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