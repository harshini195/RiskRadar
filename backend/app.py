"""
RiskRadar - Main Flask Application
"""
from flask import Flask
from flask_cors import CORS
from routes.risk_routes import risk_bp
from routes.route_routes import route_bp
from routes.hotspot_routes import hotspot_bp
from config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app)

    app.register_blueprint(risk_bp, url_prefix='/api/risk')
    app.register_blueprint(route_bp, url_prefix='/api/routes')
    app.register_blueprint(hotspot_bp, url_prefix='/api/hotspots')

    @app.route('/api/health')
    def health():
        return {'status': 'ok', 'service': 'RiskRadar API'}

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)
