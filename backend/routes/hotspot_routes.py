"""
Accident hotspot endpoints — DBSCAN clustering over accident records stored
in PostGIS, returning cluster centroids with risk classification.
"""
from flask import Blueprint, request, jsonify

hotspot_bp = Blueprint('hotspots', __name__)


# ── Mock data store (replace with PostGIS queries) ────────────────────────────
_MOCK_HOTSPOTS = [
    {'id':1,'lat':12.917,'lon':77.622,'name':'Silk Board Junction',
     'accidents':43,'risk_score':0.91,'risk_label':'High','main_cause':'Signal Violation'},
    {'id':2,'lat':12.839,'lon':77.672,'name':'Electronic City Flyover',
     'accidents':31,'risk_score':0.83,'risk_label':'High','main_cause':'Speeding'},
    {'id':3,'lat':12.951,'lon':77.591,'name':'Dairy Circle',
     'accidents':27,'risk_score':0.76,'risk_label':'High','main_cause':'Poor Visibility'},
    {'id':4,'lat':13.035,'lon':77.597,'name':'Hebbal Flyover',
     'accidents':22,'risk_score':0.69,'risk_label':'Moderate','main_cause':'Merging Traffic'},
    {'id':5,'lat':12.959,'lon':77.698,'name':'Marathahalli Bridge',
     'accidents':18,'risk_score':0.61,'risk_label':'Moderate','main_cause':'Potholes'},
    {'id':6,'lat':12.907,'lon':77.539,'name':'Mysore Road Junction',
     'accidents':15,'risk_score':0.58,'risk_label':'Moderate','main_cause':'Speeding'},
    {'id':7,'lat':13.012,'lon':77.578,'name':'Hebbal Lake Road',
     'accidents':11,'risk_score':0.44,'risk_label':'Moderate','main_cause':'Wet Roads'},
    {'id':8,'lat':12.978,'lon':77.748,'name':'Whitefield Road',
     'accidents':9,'risk_score':0.39,'risk_label':'Low','main_cause':'Potholes'},
]


@hotspot_bp.route('/', methods=['GET'])
def get_hotspots():
    """
    GET /api/hotspots/?lat=12.97&lon=77.59&radius=10&min_risk=0.4
    Returns hotspots within radius km of given point, optionally filtered by risk.
    """
    try:
        lat      = float(request.args.get('lat', 12.97))
        lon      = float(request.args.get('lon', 77.59))
        radius   = float(request.args.get('radius', 20))
        min_risk = float(request.args.get('min_risk', 0.0))
    except ValueError:
        return jsonify({'error': 'Invalid query parameters'}), 400

    # In production: PostGIS ST_DWithin query
    # SELECT * FROM hotspots
    # WHERE ST_DWithin(location, ST_MakePoint(:lon,:lat)::geography, :radius*1000)
    # AND risk_score >= :min_risk
    # ORDER BY risk_score DESC;

    def _dist_km(a_lat, a_lon, b_lat, b_lon):
        """Haversine approximation."""
        import math
        R = 6371
        dlat = math.radians(b_lat - a_lat)
        dlon = math.radians(b_lon - a_lon)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(a_lat)) * \
            math.cos(math.radians(b_lat)) * math.sin(dlon/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    filtered = [
        h for h in _MOCK_HOTSPOTS
        if _dist_km(lat, lon, h['lat'], h['lon']) <= radius
        and h['risk_score'] >= min_risk
    ]
    filtered.sort(key=lambda h: h['risk_score'], reverse=True)
    return jsonify({'hotspots': filtered, 'count': len(filtered)})


@hotspot_bp.route('/recompute', methods=['POST'])
def recompute_hotspots():
    """
    POST /api/hotspots/recompute
    Re-runs DBSCAN clustering on latest accident data.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'ml'))
    from train import generate_accident_data, detect_hotspots

    df = generate_accident_data(n=1200)
    _, hotspots = detect_hotspots(df)
    return jsonify({
        'message': f'Recomputed {len(hotspots)} hotspot clusters',
        'hotspots': hotspots[:10],
    })
