# RiskRadar 🚦

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)

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
    │  /api/routes/analyze       — fetch & score alternative routes (Google Directions)
    │  /api/risk/predict         — ML risk prediction, single segment
    │  /api/risk/predict/batch   — ML risk prediction, batch
    │  /api/risk/model-metrics   — trained model metrics (used by the Sidebar UI)
    │  /api/hotspots/            — DBSCAN accident hotspot clusters (seed + live)
    │  /api/hotspots/report      — log a new live accident report
    │  /api/hotspots/on-route    — hotspots within a buffer of a given route polyline
    ▼
ML Layer (scikit-learn + XGBoost)
    │  XGBoost (best model) / Random Forest / Gradient Boosting / Logistic Regression
    │  DBSCAN spatial clustering (seed hotspots + live in-memory accident reports)
    ▼
Storage
    │  ml/outputs/*.json + best_model.pkl — model artifacts, hotspot/locality data
    │  Hotspots are served from an in-memory cache (seeded + DBSCAN-reclustered),
    │  NOT from a live database — see "About PostgreSQL/PostGIS" below.
```

> **Note:** `ai_insights.py` is a separate explainability layer — it never predicts risk itself, it only generates the human-readable "why is this segment risky" text (title/description/advice) shown per route step.

---

## About PostgreSQL/PostGIS

`backend/schema.sql` and `docker-compose.yml` define a PostGIS-backed `accidents` / `hotspots` / `road_segments` schema, and `backend/config.py` reads a `DATABASE_URL`. **This isn't wired up in the current code path** — no route or module in `backend/` actually opens a DB connection. Hotspots are computed once from a static seed list + DBSCAN over in-memory live reports (see `backend/routes/hotspot_routes.py`). The schema exists for a future move to persistent storage; you can safely skip Postgres entirely for local development.

---

## Quick Start (Docker)

```bash
# 1. Clone and enter the project
git clone https://github.com/harshini195/RiskRadar
cd RiskRadar

# 2. Set your API key
export GOOGLE_MAPS_API_KEY=your_key_here

# 3. Build and start everything
docker compose up --build

# App: http://localhost:3000
# API: http://localhost:5000/api/health
```

The frontend build receives your key as `VITE_GMAPS_API_KEY` (Vite's required prefix), passed through as a Docker build `arg` — not a runtime `environment` var, since Vite bakes it into the static JS at build time. The backend's build context is the repo root, so `ml/` is available during the image build and `ml/train.py` runs for real (from `ml/`, so its relative `data/` path resolves correctly) — expect the first `docker compose up --build` to take a few minutes while it trains on the full dataset.

---

## Manual Setup

### 1. Train the ML model (required before starting the backend)

`ml/outputs/*.pkl` are git-ignored, so you need to generate them once locally.

```bash
cd ml

# Install ML dependencies (not in backend/requirements.txt)
pip install pandas numpy scikit-learn xgboost

# The full cleaned dataset (ml/data/cleaned_accidents_full.csv) is already
# committed, so you can train directly:
python train.py
# → writes ml/outputs/best_model.pkl, feature_columns.pkl, metrics.json,
#   locality_encodings.json, locality_cluster_map.json, hotspots.json, etc.

# Only needed if you want to regenerate the cleaned dataset from raw data:
# ml/data/AccidentReports.csv and ml/data/cleaned_accidents.csv are Git LFS
# pointers — run `git lfs pull` first, then:
#   python preprocess.py   # AccidentReports.csv -> cleaned_accidents.csv
```

`train.py` trains Random Forest, Gradient Boosting, Logistic Regression, and XGBoost, picks the best by weighted F1, and saves it as `best_model.pkl`. Current results (70/30 stratified split, `random_state=42`, 199,210 train / 85,376 test rows):

| Model | Accuracy | F1 (weighted) |
| --- | --- | --- |
| **XGBoost (best)** | **54.1%** | **0.543** |
| Random Forest | 50.1% | 0.503 |
| Gradient Boosting | 52.6% | 0.496 |
| Logistic Regression | 40.4% | 0.400 |

### 2. Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies (includes xgboost, needed to unpickle best_model.pkl)
pip install -r requirements.txt

# Set environment variables
export GOOGLE_MAPS_API_KEY=your_key_here
# DATABASE_URL is read by config.py but not currently used — safe to leave unset

# Start Flask development server
python app.py
# → Running on http://localhost:5000
```

### 3. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Set environment variables (Vite, not CRA — must be prefixed VITE_)
echo "VITE_GMAPS_API_KEY=your_key_here" >> .env.local

# Start dev server
npm run dev
# → Running on http://localhost:3000
```

The API base URL is hardcoded to `http://localhost:5000/api` in `frontend/src/utils/api.js` — no env var needed for that part locally.

---

## Google Maps API Setup

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a project → Enable these APIs:
   - **Maps JavaScript API**
   - **Directions API**
   - **Geometry library** (enabled alongside Maps JS API)
3. Create an API key under Credentials
4. Paste the key into `VITE_GMAPS_API_KEY` (frontend) and `GOOGLE_MAPS_API_KEY` (backend)

---

## ML Model

### Features used for risk prediction

Locality is treated as a first-class feature (frequency-encoded, log volume, per-locality accident counts), alongside a DBSCAN geo-cluster on lat/lon and the original road/accident attributes:

| Feature | Description |
| --- | --- |
| `locality_freq_enc`, `locality_accident_count`, `locality_log_volume` | Locality-level accident aggregates |
| `geo_cluster` | DBSCAN spatial cluster id (lat/lon) |
| `road_type_encoded` | 0=Village, 1=Other, 2=City/Town, 3=State Hwy, 4=NH/Expressway |
| `road_condition` | 0=No Defects, 1=Pot Holes, 2=Other Defects |
| `junction_control` | 0=Not at Junction, 1=Uncontrolled, 2=Signalised, 3=Roundabout |
| `weather_risk` | 0=Clear, 1=Moderate/Severe |
| `main_cause_encoded` | 0=Unknown, 1=Road Defect, 2=Human Error |
| `road_character_encoded` | 0=Straight, 1=Curve/Bend |
| `is_urban`, `is_highway`, `hit_run` | Binary flags |
| `vehicles_avg`, `log_accident_count`, `accident_sqrt`, `vehicles_log` | Vehicle/traffic volume features |
| `year_recency` | How recent the record is |
| `urban_road`, `risk_junction`, `urban_traffic`, `weather_road`, `junction_traffic`, `risk_weather_vehicle`, `urban_junction`, `traffic_intensity`, `busy_junction` | Engineered interaction features |

The full fallback list lives in `ml/predict.py::FEATURE_COLUMNS`; the actual columns used are whatever `train.py` saved to `feature_columns.pkl`.

### Output

| Field | Description |
| --- | --- |
| `risk_class` / `risk_level` | 0=Low, 1=Medium, 2=High |
| `risk_score` | Continuous score, `{0: 0.2, 1: 0.55, 2: 0.85}` per class (see `ml/predict.py::RISK_SCORE_MAP`) |
| `probabilities` | Per-class probabilities |

Note: route-level risk (`/api/routes/analyze`) buckets the *confidence-weighted average* of per-step scores into its own `Low` / `Moderate` / `High` label (thresholds 0.34 / 0.67), which is a route-scoring convention, not the model's own class names.

### Hotspot Detection (DBSCAN)

```
eps         = 0.5 km   (neighbourhood radius)
min_samples = 5        (minimum accidents to form a cluster)
```

Hotspots start from an 8-point seed list (`backend/routes/hotspot_routes.py::_SEED_HOTSPOTS`) and are re-clustered with live-reported accidents (`POST /api/hotspots/report`) at most once every 60 seconds, or immediately via `POST /api/hotspots/recompute`.

---

## API Reference

### `POST /api/routes/analyze`

```json
{ "origin": "Hebbal, Bangalore", "destination": "Varthur, Bangalore" }
```

Returns alternative routes (from Google Directions) sorted safest-first, each with a per-step `risk_trend`, flagged `route_insights`, and any hotspots on the path.

### `POST /api/risk/predict`

```json
{ "segment": { "road_type_encoded": 4, "junction_control": 2, "...": "..." } }
```

Returns `{ success, result: { risk_level, risk_label, risk_score, probabilities } }`.

### `POST /api/risk/predict/batch`

```json
{ "segments": [ { "...": "..." }, { "...": "..." } ] }
```

Returns `{ success, count, results: [...] }`.

### `GET /api/risk/model-metrics`

Returns formatted accuracy/precision/recall/F1 and feature-importance breakdown for the UI's model-metrics panel (reads `ml/outputs/metrics.json`, auto-detects whichever model `train.py` picked as best).

### `GET /api/risk/metrics`

Returns the raw contents of `ml/outputs/metrics.json` (all four models, not just the best one).

### `GET /api/hotspots/?lat=12.97&lon=77.59&radius=20&min_risk=0.4`

Returns hotspot clusters within `radius` km of the given point, filtered by `min_risk`.

### `POST /api/hotspots/report`

```json
{ "lat": 12.97, "lon": 77.59, "severity": 2, "cause": "Speeding" }
```

Logs a live accident report; folded into the next DBSCAN recompute.

### `POST /api/hotspots/on-route`

```json
{ "polyline": "<google encoded polyline>", "buffer_km": 0.3 }
```

Returns hotspots within `buffer_km` of the given route path, in driving order.

### `POST /api/hotspots/recompute`

Forces an immediate DBSCAN re-clustering of all live reports.

### `GET /api/health`

Basic liveness check.

---

## Real Accident Data Sources

The bundled dataset is Karnataka Police accident records (`ml/data/AccidentReports.csv`, tracked via Git LFS). To use other/additional real accident records:

- **India**: [data.gov.in — Road Accidents](https://data.gov.in/catalog/road-accidents-india)
- **Kaggle**: [Road Accident Severity Dataset](https://www.kaggle.com/datasets/devansodariya/road-accident-united-kingdom-uk-dataset)
- **OpenStreetMap**: Road type and junction data via Overpass API

Drop new records into `ml/data/`, update `ml/preprocess.py` and `ml/train.py`'s `DATA_PATH` as needed, and re-run `python preprocess.py && python train.py`.

---

## Project Structure

```
RiskRadar/
├── backend/
│   ├── app.py                    # Flask app factory, /api/health, /api/risk/model-metrics, /predict
│   ├── config.py                 # Configuration (GOOGLE_MAPS_API_KEY, unused DATABASE_URL)
│   ├── schema.sql                # PostgreSQL + PostGIS schema (not currently wired up)
│   ├── ai_insights.py            # Explainability layer — turns a raw segment + risk_level into
│   │                              #   human-readable title/description/advice text
│   ├── debug_junction_compare.py # Local debugging helper
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── ml/
│   │   └── generate_hotspots.py
│   ├── utils/
│   │   └── geo.py                # decode_polyline / haversine / hotspots_on_route
│   └── routes/
│       ├── risk_routes.py        # /api/risk/*
│       ├── route_routes.py       # /api/routes/*
│       └── hotspot_routes.py     # /api/hotspots/* (in-memory seed + DBSCAN live reclustering)
├── ml/
│   ├── preprocess.py             # AccidentReports.csv -> cleaned_accidents.csv
│   ├── eda.py                    # Exploratory analysis -> eda_report.txt + ml/data/plots/
│   ├── train.py                  # Trains RF/GB/LogReg/XGBoost, saves best_model.pkl + metrics.json
│   ├── predict.py                # RiskPredictor class used by the backend at inference time
│   ├── split.py                  # Standalone scratch script, not part of the live pipeline
│   ├── data/                     # AccidentReports.csv, cleaned_accidents(_full).csv (LFS)
│   └── outputs/                  # best_model.pkl, feature_columns.pkl, metrics.json, hotspots.json, etc.
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.jsx              # reads VITE_GMAPS_API_KEY
│   │   ├── components/
│   │   │   ├── MapView.jsx        # Google Maps + polylines + hotspot markers
│   │   │   ├── Sidebar.jsx        # Route planner, alerts, ML metrics panel
│   │   │   ├── RouteCard.jsx
│   │   │   └── Header.jsx
│   │   └── utils/
│   │       └── api.js             # Fetch wrappers, hardcoded BASE_URL
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   ├── Dockerfile
│   └── nginx.conf
├── outputs/                       # Top-level copies of hotspots.json / metrics.json
├── dev.bat                        # Windows helper: launches backend + frontend dev servers
└── docker-compose.yml
```

---

## Extending the Project

- **Wire up PostgreSQL/PostGIS**: `schema.sql` is ready — persist accidents/hotspots/road_segments instead of the in-memory seed + live-report cache
- **Real-time data**: Connect to traffic APIs (TomTom, HERE) to update risk scores live
- **Mobile app**: The Flask API works with any client — wrap in React Native
- **More ML models**: Add LightGBM or a neural network in `ml/train.py`
- **Alerts**: Push notifications via Firebase when users enter high-risk zones
