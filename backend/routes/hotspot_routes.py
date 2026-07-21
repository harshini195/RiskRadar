"""
Accident hotspot endpoints.

Hotspots start from a static seed list (what used to be the only data),
but the service now also accepts live accident reports and periodically
re-clusters them with DBSCAN, so the hotspot list can grow/shift as new
reports come in — without needing a database or a background scheduler.

Design (kept intentionally simple, no new infra):
  - Accident reports are appended to an in-memory list (ACCIDENT_LOG).
  - GET /api/hotspots/ recomputes the cached hotspot list at most once
    every RECOMPUTE_INTERVAL_SEC, by re-running DBSCAN over the seed
    hotspots (treated as historical clusters) plus the raw live reports.
  - A threading.Lock guards the shared state, since Flask's dev server
    can run multi-threaded and two requests could otherwise race on the
    same recompute.
"""
import math
import threading
import time
from flask import Blueprint, request, jsonify

import numpy as np
from sklearn.cluster import DBSCAN

hotspot_bp = Blueprint('hotspots', __name__)

# ── DBSCAN params (match ml/train.py's geo-clustering) ────────────────────
DBSCAN_EPS_KM     = 0.5   # 500 m neighbourhood radius
DBSCAN_MIN_SAMPLES = 5    # min accidents to form a cluster
RECOMPUTE_INTERVAL_SEC = 60  # don't re-cluster more often than this

# ── Seed hotspots (stand in for historical PostGIS data) ──────────────────
_SEED_HOTSPOTS = [
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

# ── Shared mutable state (guarded by _lock) ────────────────────────────────
_lock          = threading.Lock()
ACCIDENT_LOG   = []                 # raw live reports: [{lat, lon, timestamp, severity, cause}, ...]
_cached_hotspots = list(_SEED_HOTSPOTS)
_last_recompute  = 0.0
_next_live_id    = 10_000            # live-cluster ids start high, away from seed ids 1-8


def get_all_hotspots():
    """
    Public accessor used by route_routes.py for on-route matching.
    Triggers a recompute check first, so route analysis always sees
    reasonably fresh hotspots without callers needing to know about
    the caching mechanism.
    """
    _maybe_recompute()
    with _lock:
        return list(_cached_hotspots)


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def _maybe_recompute(force=False):
    """Recompute the cached hotspot list if the interval has elapsed."""
    global _last_recompute, _cached_hotspots, _next_live_id

    with _lock:
        now = time.time()
        if not force and (now - _last_recompute) < RECOMPUTE_INTERVAL_SEC:
            return  # cache is still fresh, nothing to do

        if not ACCIDENT_LOG:
            # No live reports yet — nothing new to cluster, keep seed list.
            _last_recompute = now
            return

        points = np.radians([[r['lat'], r['lon']] for r in ACCIDENT_LOG])
        eps_rad = DBSCAN_EPS_KM / 6371.0
        labels = DBSCAN(
            eps=eps_rad, min_samples=DBSCAN_MIN_SAMPLES,
            algorithm='ball_tree', metric='haversine',
        ).fit_predict(points)

        live_clusters = []
        for label in set(labels):
            if label == -1:
                continue  # noise, not a real cluster
            members = [ACCIDENT_LOG[i] for i in range(len(ACCIDENT_LOG)) if labels[i] == label]
            lat = sum(m['lat'] for m in members) / len(members)
            lon = sum(m['lon'] for m in members) / len(members)
            count = len(members)
            severities = [m.get('severity', 1) for m in members]
            avg_sev = sum(severities) / len(severities)
            risk_score = min(1.0, 0.3 + 0.1 * count + 0.1 * avg_sev)
            causes = [m.get('cause', 'Unknown') for m in members]
            main_cause = max(set(causes), key=causes.count)

            live_clusters.append({
                'id': _next_live_id,
                'lat': round(lat, 5),
                'lon': round(lon, 5),
                'name': f'Live cluster near ({lat:.3f}, {lon:.3f})',
                'accidents': count,
                'risk_score': round(risk_score, 2),
                'risk_label': 'High' if risk_score >= 0.7 else 'Moderate' if risk_score >= 0.4 else 'Low',
                'main_cause': main_cause,
            })
            _next_live_id += 1

        _cached_hotspots = list(_SEED_HOTSPOTS) + live_clusters
        _last_recompute = now


@hotspot_bp.route('/', methods=['GET'])
def get_hotspots():
    """
    GET /api/hotspots/?lat=12.97&lon=77.59&radius=10&min_risk=0.4
    Returns hotspots within radius km of given point, optionally filtered
    by risk. Triggers a recompute if the cache is stale (see
    RECOMPUTE_INTERVAL_SEC) so this list reflects recent live reports.
    """
    try:
        lat      = float(request.args.get('lat', 12.97))
        lon      = float(request.args.get('lon', 77.59))
        radius   = float(request.args.get('radius', 20))
        min_risk = float(request.args.get('min_risk', 0.0))
    except ValueError:
        return jsonify({'error': 'Invalid query parameters'}), 400

    all_hotspots = get_all_hotspots()

    filtered = [
        h for h in all_hotspots
        if _haversine_km(lat, lon, h['lat'], h['lon']) <= radius
        and h['risk_score'] >= min_risk
    ]
    filtered.sort(key=lambda h: h['risk_score'], reverse=True)
    return jsonify({'hotspots': filtered, 'count': len(filtered)})


@hotspot_bp.route('/report', methods=['POST'])
def report_accident():
    """
    POST /api/hotspots/report
    Body: { lat, lon, timestamp?, severity?, cause? }
    Appends a live accident report. It will be folded into the hotspot
    clusters on the next recompute (see RECOMPUTE_INTERVAL_SEC).
    """
    data = request.get_json(force=True) or {}

    try:
        lat = float(data['lat'])
        lon = float(data['lon'])
    except (KeyError, TypeError, ValueError):
        return jsonify({'error': 'lat and lon are required numeric fields'}), 400

    # Loose sanity bound around Karnataka — rejects obviously bad input
    # without hardcoding an exact boundary.
    if not (10.0 <= lat <= 19.0 and 74.0 <= lon <= 79.0):
        return jsonify({'error': 'lat/lon outside expected service area'}), 400

    report = {
        'lat': lat,
        'lon': lon,
        'timestamp': data.get('timestamp', time.time()),
        'severity': data.get('severity', 1),
        'cause': data.get('cause', 'Unknown'),
    }

    with _lock:
        ACCIDENT_LOG.append(report)

    return jsonify({'message': 'Report recorded', 'total_reports': len(ACCIDENT_LOG)}), 201


@hotspot_bp.route('/on-route', methods=['POST'])
def get_hotspots_on_route():
    """
    POST /api/hotspots/on-route
    Body: { polyline: "<google encoded polyline>", buffer_km: 0.3 }
    Returns hotspots (seed + live) within buffer_km of the given path,
    in driving order, each annotated with distance_to_route_km and
    distance_from_start_km.
    """
    from utils.geo import decode_polyline, hotspots_on_route

    data      = request.get_json(force=True) or {}
    polyline  = data.get('polyline')
    buffer_km = float(data.get('buffer_km', 0.3))

    if not polyline:
        return jsonify({'error': 'polyline is required'}), 400

    try:
        path = decode_polyline(polyline)
    except Exception:
        return jsonify({'error': 'Invalid polyline'}), 400

    matched = hotspots_on_route(path, get_all_hotspots(), buffer_km=buffer_km)
    return jsonify({'hotspots': matched, 'count': len(matched)})


@hotspot_bp.route('/recompute', methods=['POST'])
def recompute_hotspots():
    """
    POST /api/hotspots/recompute
    Forces an immediate DBSCAN re-clustering of all live reports,
    ignoring the normal 60s cache interval.
    """
    _maybe_recompute(force=True)
    hotspots = get_all_hotspots()
    return jsonify({
        'message': f'Recomputed — {len(hotspots)} hotspots total '
                   f'({len(ACCIDENT_LOG)} live reports on file)',
        'hotspots': hotspots,
    })