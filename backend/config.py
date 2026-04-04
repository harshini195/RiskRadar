import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'riskradar-dev-key')
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        'postgresql://postgres:password@localhost:5432/riskradar'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    GOOGLE_MAPS_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY', '')
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml', 'risk_model.pkl')
    SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml', 'scaler.pkl')