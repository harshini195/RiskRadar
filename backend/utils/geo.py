"""
Geometry helpers for matching accident hotspots against a driving route.

- decode_polyline()      : Google encoded-polyline -> list[(lat, lng)]
- haversine_km()         : great-circle distance between two points
- point_to_segment_km()  : min distance from a point to a line segment
- hotspots_on_route()    : which hotspots fall within a buffer of a path,
                            annotated with distance-to-route and how far
                            along the route (from the origin) they sit
"""
import math


def decode_polyline(polyline_str):
    """Decode a Google Maps encoded polyline string into [(lat, lng), ...]."""
    index, lat, lng = 0, 0, 0
    coordinates = []
    length = len(polyline_str)

    while index < length:
        for unit in ('lat', 'lng'):
            shift, result = 0, 0
            while True:
                b = ord(polyline_str[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if (result & 1) else (result >> 1)
            if unit == 'lat':
                lat += delta
            else:
                lng += delta
        coordinates.append((lat / 1e5, lng / 1e5))

    return coordinates


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in km."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def point_to_segment_km(p, a, b):
    """
    Approx min distance (km) from point p to segment a-b, and the nearest
    point on that segment (lat, lng).
    Uses a local equirectangular (flat-earth) projection centred on the
    segment — accurate at road/city scale (errors are negligible under
    a few hundred km).
    p, a, b are (lat, lng) tuples.
    Returns (distance_km, nearest_point) where nearest_point is (lat, lng).
    """
    lat0 = math.radians((a[0] + b[0]) / 2)
    kx = 111.32 * math.cos(lat0)  # km per degree of longitude at this latitude
    ky = 110.57                   # km per degree of latitude (~constant)

    def to_xy(pt):
        return pt[1] * kx, pt[0] * ky

    ax, ay = to_xy(a)
    bx, by = to_xy(b)
    px, py = to_xy(p)

    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 == 0:
        t = 0.0
    else:
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len2))
    cx, cy = ax + t * dx, ay + t * dy
    dist = math.hypot(px - cx, py - cy)

    # Convert the nearest point back from local xy to (lat, lng)
    nearest_lat = a[0] + t * (b[0] - a[0])
    nearest_lng = a[1] + t * (b[1] - a[1])
    return dist, (nearest_lat, nearest_lng)


def hotspots_on_route(path, hotspots, buffer_km=0.3):
    """
    Return the subset of `hotspots` that lie within `buffer_km` of `path`
    (a list of (lat, lng) points, e.g. a decoded route polyline).

    Each matched hotspot is annotated with:
      - distance_to_route_km   : how far off the route it sits
      - distance_from_start_km : progress along the route where it's closest
                                  (useful for ordering / "in 3.2 km" alerts)
      - route_lat / route_lon  : the nearest point ON the route itself —
                                  use this (not lat/lon) when drawing a
                                  marker, so it visually sits on the road
                                  even though the hotspot's real recorded
                                  location may be a short distance off it

    Results are sorted by distance_from_start_km (i.e. in driving order).
    """
    if not path or len(path) < 2 or not hotspots:
        return []

    # Cumulative distance along the path, so we can report "how far in".
    cum_dist = [0.0]
    for i in range(1, len(path)):
        cum_dist.append(cum_dist[-1] + haversine_km(*path[i - 1], *path[i]))

    matched = []
    for h in hotspots:
        point = (h['lat'], h['lon'])
        best_dist = float('inf')
        best_progress = 0.0
        best_snap = point

        for i in range(1, len(path)):
            d, snap = point_to_segment_km(point, path[i - 1], path[i])
            if d < best_dist:
                best_dist = d
                best_progress = cum_dist[i - 1]
                best_snap = snap

        if best_dist <= buffer_km:
            matched.append({
                **h,
                'distance_to_route_km': round(best_dist, 3),
                'distance_from_start_km': round(best_progress, 1),
                'route_lat': round(best_snap[0], 6),
                'route_lon': round(best_snap[1], 6),
            })

    matched.sort(key=lambda h: h['distance_from_start_km'])
    return matched