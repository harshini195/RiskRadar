"""
Route analysis — fetches alternatives from Google Maps Directions API,
scores each segment against the ML model, and returns ranked safe routes.
"""
from flask import Blueprint, request, jsonify, current_app
import requests

route_bp = Blueprint('routes', __name__)

GMAPS_DIRECTIONS = 'https://maps.googleapis.com/maps/api/directions/json'


def _score_route(legs: list, predictor) -> float:
    """
    Derive a route-level risk score by sampling road segments from
    the Directions API response steps and averaging predicted risk scores.
    In production, look up each segment's historical accident data from DB.
    Here we use a simple heuristic from step metadata as a demo.
    """
    scores = []
    for leg in legs:
        for step in leg.get('steps', []):
            maneuver = step.get('maneuver', '')
            duration = step.get('duration', {}).get('value', 30)
            distance = step.get('distance', {}).get('value', 100)

            # Heuristic feature proxy (replace with real DB lookup)
            junction_risk = 1 if 'turn' in maneuver or 'merge' in maneuver else 0
            speed_proxy   = min(100, int((distance / max(duration, 1)) * 3.6))

            segment = {
                'accident_count_6mo': 5,   # placeholder — use PostGIS query
                'severity_avg':       1.5,
                'road_type_encoded':  1,
                'road_condition':     1,
                'junction_control':   junction_risk * 2,
                'weather_risk':       0,
                'vehicles_avg':       2,
                'speed_limit':        speed_proxy,
            }
            result = predictor.predict(segment)
            scores.append(result['risk_score'])

    return round(sum(scores) / len(scores), 3) if scores else 0.5


@route_bp.route('/analyze', methods=['GET','POST'])
def analyze_routes():
    """
    POST /api/routes/analyze
    Body: { origin: "...", destination: "...", alternatives: true }
    Returns ranked routes with risk scores.
    """
    data = request.get_json(force=True)
    origin      = data.get('origin', '')
    destination = data.get('destination', '')
    if not origin or not destination:
        return jsonify({'error': 'origin and destination required'}), 400

    api_key = current_app.config['GOOGLE_MAPS_API_KEY']

    # ── Fetch routes from Google Maps ─────────────────────────────────────
    params = {
        'origin':       origin,
        'destination':  destination,
        'alternatives': 'true',
        'key':          api_key,
    }
    gmaps_resp = requests.get(GMAPS_DIRECTIONS, params=params, timeout=10)
    gmaps_data  = gmaps_resp.json()

    if gmaps_data.get('status') != 'OK':
        return jsonify({'error': 'Google Maps API error', 'details': gmaps_data.get('status')}), 502

    from routes.risk_routes import get_predictor
    predictor = get_predictor()

    # ── Score each route ──────────────────────────────────────────────────
    routes_out = []
    for i, route in enumerate(gmaps_data['routes']):
        legs     = route.get('legs', [])
        risk     = _score_route(legs, predictor)
        distance = sum(leg['distance']['value'] for leg in legs)
        duration = sum(leg['duration']['value'] for leg in legs)

        risk_label = 'High' if risk >= 0.7 else 'Moderate' if risk >= 0.4 else 'Low'

        routes_out.append({
            'route_index':  i,
            'summary':      route.get('summary', f'Route {i+1}'),
            'distance_m':   distance,
            'distance_km':  round(distance / 1000, 1),
            'duration_sec': duration,
            'duration_min': round(duration / 60, 1),
            'risk_score':   risk,
            'risk_label':   risk_label,
            'polyline':     route['overview_polyline']['points'],
            'warnings':     route.get('warnings', []),
            'copyrights':   route.get('copyrights', ''),
        })

    # Sort safest first
    routes_out.sort(key=lambda r: r['risk_score'])
    routes_out[0]['recommended'] = True

    return jsonify({
        'origin':      origin,
        'destination': destination,
        'routes':      routes_out,
        'count':       len(routes_out),
    })
