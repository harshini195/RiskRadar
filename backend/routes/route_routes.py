from flask import Blueprint, request, jsonify, current_app
import requests

route_bp = Blueprint('routes', __name__)

GMAPS_DIRECTIONS = 'https://maps.googleapis.com/maps/api/directions/json'


def _score_route(legs: list, predictor) -> float:
    scores = []
    for leg in legs:
        for step in leg.get('steps', []):
            maneuver   = step.get('maneuver', '')
            duration   = step.get('duration', {}).get('value', 30)
            distance   = step.get('distance', {}).get('value', 100)
            html_instr = step.get('html_instructions', '').lower()

            # ── Derive Road_Type from instruction text ─────────────────
            if any(x in html_instr for x in ['nh', 'national highway', 'expressway']):
                road_type = 'NH'
            elif any(x in html_instr for x in ['state highway', 'sh']):
                road_type = 'State Highway'
            elif any(x in html_instr for x in ['city', 'town', 'urban']):
                road_type = 'City or Town Road'
            elif any(x in html_instr for x in ['village', 'rural']):
                road_type = 'Village Road'
            else:
                road_type = 'City or Town Road'

            # ── Derive Collision_Type from maneuver ────────────────────
            if 'turn' in maneuver:
                collision_type = 'Right Turn'
            elif 'merge' in maneuver or 'fork' in maneuver:
                collision_type = 'Head on'
            elif 'roundabout' in maneuver:
                collision_type = 'Right Turn'
            else:
                collision_type = 'Rear end'

            # ── Junction Control from maneuver/instructions ────────────
            if any(x in html_instr for x in ['signal', 'traffic light']):
                junction_control = 'Signalised Junction'
            elif any(x in html_instr for x in ['roundabout']):
                junction_control = 'Roundabout'
            elif any(x in maneuver for x in ['turn', 'merge', 'fork']):
                junction_control = 'Uncontrolled'
            else:
                junction_control = 'Not at Junction'

            # ── Road Character ─────────────────────────────────────────
            if any(x in html_instr for x in ['straight', 'continue']):
                road_character = 'Straight Road'
            elif any(x in html_instr for x in ['curve', 'bend']):
                road_character = 'Curve'
            else:
                road_character = 'Straight Road'

            # ── Speed proxy → number of vehicles proxy ─────────────────
            speed_kmh = min(100, int((distance / max(duration, 1)) * 3.6))
            num_vehicles = 2 if speed_kmh < 30 else 3 if speed_kmh < 60 else 4

            # ── Build segment with correct feature names ───────────────
            segment = {
                'Main_Cause':          'Human Error',
                'Weather':             'Clear',
                'Road_Type':           road_type,
                'Collision_Type':      collision_type,
                'Road_Character':      road_character,
                'Surface_Condition':   'Dry',
                'Road_Condition':      'No Defects',
                'Junction_Control':    junction_control,
                'Noofvehicle_involved': num_vehicles,
                'Hit_Run':             'No',
                'Lane_Type':           'Two-Way',
                'Year':                2024,
                'DISTRICTNAME':        'Bangalore Urban',  # default; override if known
            }

            result = predictor.predict(segment)

            # Convert severity index (0-3) to a 0-1 risk score
            severity_index = result.get('severity_index', 1)
            risk_score = severity_index / 3.0
            scores.append(risk_score)

    if not scores:
        return 0.5
    return round(sum(scores) / len(scores), 3)


@route_bp.route('/analyze', methods=['POST'])
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

    # ── Fetch routes from Google Maps ──────────────────────────────────
    params = {
        'origin':       origin,
        'destination':  destination,
        'alternatives': 'true',
        'key':          api_key,
    }
    gmaps_resp = requests.get(GMAPS_DIRECTIONS, params=params, timeout=10)
    gmaps_data = gmaps_resp.json()

    print("Google Maps response:", gmaps_data.get('status'))
    print("Error message:", gmaps_data.get('error_message', 'none'))

    if gmaps_data.get('status') != 'OK':
        return jsonify({
            'error':   'Google Maps API error',
            'details': gmaps_data.get('status'),
            'message': gmaps_data.get('error_message', '')
        }), 502

    from routes.risk_routes import get_predictor
    predictor = get_predictor()

    # ── Score each route ───────────────────────────────────────────────
    routes_out = []
    for i, route in enumerate(gmaps_data['routes']):
        legs     = route.get('legs', [])
        risk     = _score_route(legs, predictor)
        distance = sum(leg['distance']['value'] for leg in legs)
        duration = sum(leg['duration']['value'] for leg in legs)

        risk_label = 'High' if risk >= 0.67 else 'Moderate' if risk >= 0.34 else 'Low'

        routes_out.append({
            'route_index':  i,
            'summary':      route.get('summary', f'Route {i + 1}'),
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