# RiskRadar 🚦
### Intelligent Accident Risk Detection for Safer Navigation

RiskRadar is a full-stack AI-powered navigation system that identifies accident-prone zones using machine learning and recommends the safest route between two locations.

---

## Architecture

```
Frontend (React + Vite)
    │  Google Maps JS API (map + routes display)
    │  Web Speech API (voice alerts)
    ▼
Backend (Python Flask)
    │  /api/routes/analyze   — fetch & score alternative routes
    │  /api/risk/predict     — ML risk prediction per segment
    │  /api/hotspots/        — DBSCAN accident hotspot clusters
    ▼
ML Layer (scikit-learn)
    │  Random Forest / Gradient Boosting / Logistic Regression
    │  DBSCAN spatial clustering
    ▼
Database (PostgreSQL + PostGIS)
    │  accidents table  — raw incident records with lat/lon
    │  hotspots table   — computed cluster centroids
    │  road_segments    — scored road segments
```

---

## Quick Start (Docker)

```bash
# 1. Clone and enter the project
git clone https://github.com/yourname/riskradar
cd riskradar

# 2. Set your API keys
cp .env.example .env
# Edit .env — add your GOOGLE_MAPS_API_KEY

# 3. Build and start everything
docker compose up --build

# App: http://localhost:3000
# API: http://localhost:5000/api/health
```

---

## Manual Setup

### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GOOGLE_MAPS_API_KEY=your_key_here
export DATABASE_URL=postgresql://postgres:password@localhost:5432/riskradar

# Set up database (requires PostgreSQL + PostGIS running)
psql -U postgres -c "CREATE DATABASE riskradar;"
psql -U postgres -d riskradar -f schema.sql

# Train the ML model (generates risk_model.pkl and scaler.pkl)
python ml/train.py

# Start Flask development server
python app.py
# → Running on http://localhost:5000
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Set environment variables
echo "REACT_APP_API_URL=http://localhost:5000/api" >> .env.local
echo "REACT_APP_GMAPS_API_KEY=your_key_here"      >> .env.local

# Start dev server
npm run dev
# → Running on http://localhost:3000
```

---

## Google Maps API Setup

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a project → Enable these APIs:
   - **Maps JavaScript API**
   - **Directions API**
   - **Geometry library** (enabled alongside Maps JS API)
3. Create an API key under Credentials
4. Paste the key into your `.env` file

---

## ML Model

### Features used for risk prediction

| Feature | Description |
|---|---|
| `accident_count_6mo` | Historical accidents in the last 6 months |
| `severity_avg` | Average accident severity (1–3) |
| `road_type_encoded` | 0=residential, 1=arterial, 2=highway |
| `road_condition` | 0=poor, 1=average, 2=good |
| `junction_control` | 0=none, 1=sign, 2=signal, 3=roundabout |
| `weather_risk` | 0=clear, 1=rain or fog |
| `vehicles_avg` | Average vehicles involved per accident |
| `speed_limit` | Posted speed limit (km/h) |

### Output

| Field | Description |
|---|---|
| `risk_class` | 0=Low, 1=Moderate, 2=High |
| `risk_score` | Continuous score 0.0–1.0 |
| `probabilities` | Per-class probabilities |

### Hotspot Detection (DBSCAN)

```
eps     = 0.5 km   (neighbourhood radius)
min_samples = 5    (minimum accidents to form a cluster)
```

Clusters are recomputed via `POST /api/hotspots/recompute`.

---

## API Reference

### `POST /api/routes/analyze`
```json
{ "origin": "Koramangala, Bangalore", "destination": "Electronic City, Bangalore" }
```
Returns alternative routes sorted by risk score (safest first).

### `POST /api/risk/predict`
```json
{ "segment": { "accident_count_6mo": 12, "severity_avg": 1.8, ... } }
```
Returns `{ risk_class, risk_score, risk_label, probabilities }`.

### `GET /api/hotspots/?lat=12.97&lon=77.59&radius=20&min_risk=0.4`
Returns hotspot clusters near the given coordinates.

### `GET /api/risk/metrics`
Returns trained model evaluation metrics (accuracy, precision, recall, F1).

---

## Real Accident Data Sources

To replace synthetic data with real accident records:

- **India**: [data.gov.in — Road Accidents](https://data.gov.in/catalog/road-accidents-india)
- **Kaggle**: [Road Accident Severity Dataset](https://www.kaggle.com/datasets/devansodariya/road-accident-united-kingdom-uk-dataset)
- **OpenStreetMap**: Road type and junction data via Overpass API

Load into `accidents` table and re-run `python ml/train.py`.

---

## Project Structure

```
riskradar/
├── backend/
│   ├── app.py                 # Flask app factory
│   ├── config.py              # Configuration
│   ├── schema.sql             # PostgreSQL + PostGIS schema
│   ├── requirements.txt
│   ├── Dockerfile
│   └── routes/
│       ├── risk_routes.py     # /api/risk/*
│       ├── route_routes.py    # /api/routes/*
│       └── hotspot_routes.py  # /api/hotspots/*
├── ml/
│   └── train.py               # Training pipeline + RiskPredictor class
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.jsx
│   │   ├── components/
│   │   │   ├── MapView.jsx    # Google Maps + polylines + hotspot markers
│   │   │   ├── Sidebar.jsx    # Route planner, alerts, ML metrics
│   │   │   ├── RouteCard.jsx
│   │   │   └── Header.jsx
│   │   └── utils/
│   │       └── api.js         # Fetch wrappers for Flask API
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   ├── Dockerfile
│   └── nginx.conf
└── docker-compose.yml
```

---

## Extending the Project

- **Real-time data**: Connect to traffic APIs (TomTom, HERE) to update risk scores live
- **Mobile app**: The Flask API works with any client — wrap in React Native
- **More ML models**: Add XGBoost, LightGBM, or a neural network in `train.py`
- **Alerts**: Push notifications via Firebase when users enter high-risk zones
- **Admin dashboard**: Add a `/admin` React page to visualise model retraining history
