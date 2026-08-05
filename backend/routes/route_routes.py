from flask import Blueprint, request, jsonify, current_app
import requests
import datetime
import math
from routes.risk_routes import get_predictor
from routes.hotspot_routes import get_all_hotspots
from utils.geo import decode_polyline, hotspots_on_route
from ai_insights import explain_segment, is_important_segment
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

def _nearby_accident_count(lat, lng, hotspots, max_km=1.0):
    """
    Find the closest known hotspot within max_km of this point and
    return its real accident count — so the 'historical accident
    hotspot' explanation is backed by real data instead of always
    using the dataset-median placeholder (8).
    """
    best_accidents = None
    best_dist = max_km
    for h in hotspots:
        d = math.radians(0)  # noop, keeps math import obviously used
        dist = _haversine_km(lat, lng, h['lat'], h['lon'])
        if dist <= best_dist:
            best_dist = dist
            best_accidents = h.get('accidents', 8)
    return best_accidents if best_accidents is not None else 8


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


# Weight used to turn the model's per-class probabilities into a single
# continuous 0-100 score, e.g. 70% Medium + 30% High != a flat bucket.
_CLASS_WEIGHT_PCT = {"Low": 0, "Medium": 50, "High": 100}


def _confidence_weighted_risk_pct(result: dict) -> float:
    """
    Turn a predictor result into a continuous 0-100 risk percentage using
    the model's real class probabilities, instead of collapsing to a flat
    0 / 50 / 100 bucket based on risk_level alone. Falls back to the
    bucketed value only if the model didn't return probabilities.
    """
    probs = result.get('probabilities')
    if probs:
        return sum(probs.get(label, 0) * weight for label, weight in _CLASS_WEIGHT_PCT.items())
    risk_level = result.get('risk_level', 1)
    return (risk_level / 2.0) * 100


def _derive_raw_segment(step: dict) -> dict:
    maneuver   = step.get('maneuver', '')
    duration   = step.get('duration', {}).get('value', 30)
    distance   = step.get('distance', {}).get('value', 100)
    html_instr = step.get('html_instructions', '').lower()
    speed_kmh  = min(120, int((distance / max(duration, 1)) * 3.6))

    if any(x in html_instr for x in ['nh', 'national highway', 'expressway']):
        road_type_encoded = 4; is_urban = 0
    elif any(x in html_instr for x in ['state highway', ' sh ']):
        road_type_encoded = 3; is_urban = 0
    elif any(x in html_instr for x in ['city', 'town', 'urban']):
        road_type_encoded = 2; is_urban = 1
    elif any(x in html_instr for x in ['village', 'rural']):
        road_type_encoded = 0; is_urban = 0
    else:
        road_type_encoded = 2; is_urban = 1

    # FIXED: only real junctions count — ordinary 'turn-left'/'turn-right'
    # maneuvers no longer trigger junction_control=1 for almost every step.
    if 'roundabout' in maneuver or 'roundabout' in html_instr:
        junction_control = 3
    elif any(x in html_instr for x in ['signal', 'traffic light']):
        junction_control = 2
    elif maneuver in ('merge', 'fork-left', 'fork-right', 'ramp-left', 'ramp-right') \
            or maneuver.startswith('uturn'):
        junction_control = 1
    else:
        junction_control = 0

    # FIXED: sharp/slight turn maneuvers now feed curve detection instead
    # of being silently absorbed into junction_control.
    if any(x in html_instr for x in ['curve', 'bend', 'winding']) \
            or maneuver.startswith('turn-slight') or maneuver.startswith('turn-sharp'):
        road_character_encoded = 1
    else:
        road_character_encoded = 0

    vehicles_avg = 1 if speed_kmh < 30 else 2 if speed_kmh < 60 else 3

    return {
        "severity_numeric":       1,
        "road_type_encoded":      road_type_encoded,
        "road_condition":         0,
        "junction_control":       junction_control,
        "weather_risk":           0,
        "vehicles_avg":           vehicles_avg,
        "main_cause_encoded":     2,
        "hit_run":                0,
        "road_character_encoded": road_character_encoded,
        "is_urban":               is_urban,
        "accident_count_6mo":     8,
        "Year":                   CURRENT_YEAR,
        # NEW: needed for phrasing variety in ai_insights.py
        "lat": step["start_location"]["lat"],
        "lng": step["start_location"]["lng"],
    }

def _segment_reason(raw, risk_level):
    reasons = []

    if raw["road_type_encoded"] == 4:
        reasons.append("Busy highway")

    if raw["junction_control"] == 3:
        reasons.append("Roundabout ahead")

    elif raw["junction_control"] == 2:
        reasons.append("Traffic signal")

    elif raw["junction_control"] == 1:
        reasons.append("Busy intersection")

    if raw["road_character_encoded"] == 1:
        reasons.append("Sharp curve")

    if risk_level == 2:
        reasons.append("Higher accident probability")

    if not reasons:
        reasons.append("Normal road conditions")

    return reasons

def _score_route(legs: list, predictor, hotspots) -> float:
    from predict import RiskPredictor
    hotspots = hotspots or []
    scores = []
    risk_trend = []
    route_insights = []
    distance_so_far = 0.0
    step_num = 0
    recent_factors = []
    hotspots = hotspots or []

    for leg in legs:
        for step in leg.get('steps', []):
            step_num += 1
            raw = _derive_raw_segment(step)

            # NEW: override the placeholder with a real nearby accident
            # count, if one exists within 1km of this step.
            raw["accident_count_6mo"] = _nearby_accident_count(
                step["start_location"]["lat"],
                step["start_location"]["lng"],
                hotspots,
            )

            segment = RiskPredictor.engineer(raw)
            result  = predictor.predict(segment)
            risk_level = result.get('risk_level', 1)
            risk_pct = _confidence_weighted_risk_pct(result)
            scores.append(risk_pct / 100.0)
            step_distance = step["distance"]["value"] / 1000
            distance_so_far += step_distance

            insight = explain_segment(raw, risk_level, recent_factors)
            if insight.get("factor"):
                recent_factors.append(insight["factor"])
                recent_factors = recent_factors[-3:]


            risk_trend.append({
                "lat": step["start_location"]["lat"],
                "lng": step["start_location"]["lng"],
                "distance": round(distance_so_far, 2),
                "risk": round(risk_pct, 1),
                "title": insight["title"],
                "description": insight["description"],
                "advice": insight["advice"],
            })

            if is_important_segment(raw, risk_level, insight):   # CHANGED
                route_insights.append({
                    "lat": step["start_location"]["lat"],
                    "lng": step["start_location"]["lng"],
                    "distance": round(distance_so_far, 2),
                    "risk": round(risk_pct, 1),
                    "title": insight["title"],
                    "description": insight["description"],
                    "advice": insight["advice"],
                })

            instr = step.get('html_instructions', '')[:50]
            level_name = ['Low', 'Moderate', 'High'][risk_level]
            print(f"    step {step_num:>2}: \"{instr}...\" -> "
                  f"road_type={raw['road_type_encoded']} junction={raw['junction_control']} "
                  f"curve={raw['road_character_encoded']} -> predicted={level_name} ({risk_pct:.1f}%) "
                  f"factor={insight.get('factor')} conf={insight.get('confidence')}")

    final_score = round(sum(scores) / len(scores), 3) if scores else 0.5
    print(f"    -> route risk_score = average of {len(scores)} steps = {final_score}")
    return final_score, risk_trend, route_insights


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
    print("Google raw route count:", len(gmaps_data['routes']))

    if gmaps_data.get('status') != 'OK':
        return jsonify({'error': 'Google Maps API error',
                        'details': gmaps_data.get('status'),
                        'message': gmaps_data.get('error_message', '')}), 502

    predictor    = get_predictor()
    all_hotspots = get_all_hotspots()

    routes_out = []
    for i, route in enumerate(gmaps_data['routes']):
        legs     = route.get('legs', [])
        risk, risk_trend, route_insights = _score_route(legs, predictor, all_hotspots)
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
            'risk_trend':          risk_trend,
            'route_insights':      route_insights,
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