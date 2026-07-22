from flask import Blueprint, request, jsonify, current_app
import requests
import datetime
import math
from routes.risk_routes import get_predictor
from routes.hotspot_routes import get_all_hotspots
from utils.geo import decode_polyline, hotspots_on_route
route_bp = Blueprint('routes', __name__)
GMAPS_DIRECTIONS = 'https://maps.googleapis.com/maps/api/directions/json'

# Distance (km) within which a known hotspot is considered "on" a route.
HOTSPOT_BUFFER_KM = 0.3

# ── Encoding maps ─────────────────────────────────────────────────────────────
# road_type_encoded : 0=Village, 1=Other, 2=City/Town, 3=State Hwy, 4=NH/Expressway, 5=Other
# road_condition    : 0=No Defects, 1=Pot Holes, 2=Other Defects
# junction_control  : 0=Not at Junction, 1=Uncontrolled, 2=Signalised, 3=Roundabout
# weather_risk      : 0=Clear, 1=Moderate/Severe
# main_cause_encoded: 0=Unknown, 1=Road Defect, 2=Human Error
# road_character_enc: 0=Straight, 1=Curve/Bend
# severity_numeric  : 0=Damage Only, 1=Simple Injury, 2=Grievous Injury, 3=Fatal
#   → for route steps we use 1 (Simple Injury) as a neutral default;
#     override with live data if available.
# ─────────────────────────────────────────────────────────────────────────────

CURRENT_YEAR = datetime.datetime.now().year


def _derive_raw_segment(step: dict) -> dict:
    """
    Map one Google Maps step → raw segment dict with base fields.
    Engineered features (log_accident_count, year_recency, etc.) are
    added by RiskPredictor.engineer() before prediction.
    """
    maneuver   = step.get('maneuver', '')
    duration   = step.get('duration', {}).get('value', 30)   # seconds
    distance   = step.get('distance', {}).get('value', 100)  # metres
    html_instr = step.get('html_instructions', '').lower()
    speed_kmh  = min(120, int((distance / max(duration, 1)) * 3.6))

    # road_type_encoded + helper flags
    if any(x in html_instr for x in ['nh', 'national highway', 'expressway']):
        road_type_encoded = 4; is_urban = 0
    elif any(x in html_instr for x in ['state highway', ' sh ']):
        road_type_encoded = 3; is_urban = 0
    elif any(x in html_instr for x in ['city', 'town', 'urban']):
        road_type_encoded = 2; is_urban = 1
    elif any(x in html_instr for x in ['village', 'rural']):
        road_type_encoded = 0; is_urban = 0
    else:
        road_type_encoded = 2; is_urban = 1  # default: city road

    # junction_control
    if any(x in html_instr for x in ['signal', 'traffic light']):
        junction_control = 2
    elif 'roundabout' in html_instr or 'roundabout' in maneuver:
        junction_control = 3
    elif any(x in maneuver for x in ['turn', 'merge', 'fork']):
        junction_control = 1
    else:
        junction_control = 0

    # road_character_encoded
    if any(x in html_instr for x in ['curve', 'bend', 'winding']):
        road_character_encoded = 1
    else:
        road_character_encoded = 0

    # vehicles_avg proxy from speed
    vehicles_avg = 1 if speed_kmh < 30 else 2 if speed_kmh < 60 else 3

    return {
        # Key signal — neutral default; override with DB lookup if available
        "severity_numeric":      1,

        # Road features derived from step
        "road_type_encoded":     road_type_encoded,
        "road_condition":        0,   # No defects (default)
        "junction_control":      junction_control,
        "weather_risk":          0,   # Clear (default; override from weather API)
        "vehicles_avg":          vehicles_avg,
        "main_cause_encoded":    2,   # Human Error (most common)
        "hit_run":               0,
        "road_character_encoded": road_character_encoded,
        "is_urban":              is_urban,

        # For engineer()
        "accident_count_6mo":    8,   # dataset median fallback
        "Year":                  CURRENT_YEAR,
    }


def _score_route(legs: list, predictor) -> float:
    from predict import RiskPredictor  # for engineer() static method
    scores = []
    step_num = 0
    for leg in legs:
        for step in leg.get('steps', []):
            step_num += 1
            raw     = _derive_raw_segment(step)
            segment = RiskPredictor.engineer(raw)
            result  = predictor.predict(segment)
            risk_level = result.get('risk_level', 1)
            scores.append(risk_level / 2.0)

            # Debug: show exactly what this step was guessed as, and what
            # the model predicted for it, so the final average is fully
            # traceable rather than a black box.
            instr = step.get('html_instructions', '')[:50]
            level_name = ['Low', 'Moderate', 'High'][risk_level]
            print(f"    step {step_num:>2}: \"{instr}...\" -> "
                  f"road_type={raw['road_type_encoded']} junction={raw['junction_control']} "
                  f"curve={raw['road_character_encoded']} -> predicted={level_name}")

    final_score = round(sum(scores) / len(scores), 3) if scores else 0.5
    print(f"    -> route risk_score = average of {len(scores)} steps = {final_score}")
    return final_score


@route_bp.route('/analyze', methods=['POST'])
def analyze_routes():
    """
    POST /api/routes/analyze
    Body: { origin: "...", destination: "...", alternatives: true }
    Returns routes ranked safest-first with risk scores.
    """
    data        = request.get_json(force=True)
    origin      = data.get('origin', '')
    destination = data.get('destination', '')

    if not origin or not destination:
        return jsonify({'error': 'origin and destination required'}), 400

    api_key = current_app.config['GOOGLE_MAPS_API_KEY']
    params  = {'origin': origin, 'destination': destination,
               'alternatives': 'true', 'key': api_key}

    gmaps_resp = requests.get(GMAPS_DIRECTIONS, params=params, timeout=10)
    gmaps_data = gmaps_resp.json()
    print("Google Maps status:", gmaps_data.get('status'))

    if gmaps_data.get('status') != 'OK':
        return jsonify({'error': 'Google Maps API error',
                        'details': gmaps_data.get('status'),
                        'message': gmaps_data.get('error_message', '')}), 502

    predictor    = get_predictor()
    all_hotspots = get_all_hotspots()

    routes_out = []
    for i, route in enumerate(gmaps_data['routes']):
        legs     = route.get('legs', [])
        risk     = _score_route(legs, predictor)
        distance = sum(leg['distance']['value'] for leg in legs)
        duration = sum(leg['duration']['value'] for leg in legs)

        # ── Hotspot detection along this specific route ──────────────────
        encoded_polyline = route['overview_polyline']['points']
        path              = decode_polyline(encoded_polyline)
        route_hotspots    = hotspots_on_route(path, all_hotspots, buffer_km=HOTSPOT_BUFFER_KM)
        has_high_hotspot  = any(h['risk_score'] >= 0.7 for h in route_hotspots)
        has_mod_hotspot   = any(h['risk_score'] >= 0.4 for h in route_hotspots)

        # ML risk label, nudged up if a known high-risk hotspot sits on the
        # path — the model scores road *segments*, not fixed known blackspots,
        # so this makes sure a hard-known danger point can't be missed.
        risk_label = 'High' if risk >= 0.67 else 'Moderate' if risk >= 0.34 else 'Low'
        if has_high_hotspot and risk_label != 'High':
            risk_label = 'High'
        elif has_mod_hotspot and risk_label == 'Low':
            risk_label = 'Moderate'

        routes_out.append({
            'route_index':         i,
            'summary':             route.get('summary', f'Route {i+1}'),
            'distance_m':          distance,
            'distance_km':         round(distance / 1000, 1),
            'duration_sec':        duration,
            'duration_min':        round(duration / 60, 1),
            'risk_score':          risk,
            'risk_label':          risk_label,
            'polyline':            encoded_polyline,
            'warnings':            route.get('warnings', []),
            'copyrights':          route.get('copyrights', ''),
            'hotspots_on_route':   route_hotspots,
            'hotspot_count':       len(route_hotspots),
            'has_high_risk_hotspot': has_high_hotspot,
        })

    # Sort safest-first: lowest risk score wins, then fewest hotspots, then
    # (when those tie, as with routes that round to the same risk bucket)
    # fastest route wins — so equally-risky options don't sort arbitrarily.
    routes_out.sort(key=lambda r: (r['risk_score'], r['hotspot_count'], r['duration_sec']))
    if routes_out:
        routes_out[0]['recommended'] = True

    return jsonify({'origin': origin, 'destination': destination,
                    'routes': routes_out, 'count': len(routes_out)})