"""
ai_insights.py
----------------
Explainability layer for RiskRadar.

This module NEVER predicts risk — it only explains WHY a segment
was flagged, using a scored multi-factor detector system instead
of a fixed if/elif priority chain (which was the root cause of
repetitive explanations).
"""

import hashlib

# ── Helper: deterministic phrasing variety (not random — same segment
#    always gets the same phrasing, but different segments vary) ──────────
def _variant_index(n, *seed_parts):
    h = hashlib.md5("|".join(str(p) for p in seed_parts).encode()).hexdigest()
    return int(h, 16) % n


# ── Risk factor detectors ──────────────────────────────────────────────────
# Each returns a confidence score 0.0–1.0. 0.0 = "doesn't apply here".
# Confidence is relative to how strongly the evidence points to that
# factor — NOT a fixed priority order. This is what fixes the repetition:
# a near-constant feature just never scores high, instead of dominating
# because it happened to be checked first.

def _highway(raw):
    if raw.get("road_type_encoded") == 4:
        return 0.9
    if raw.get("road_type_encoded") == 3:
        return 0.6
    return 0.0

def _roundabout(raw):
    return 0.95 if raw.get("junction_control") == 3 else 0.0

def _signal(raw):
    return 0.7 if raw.get("junction_control") == 2 else 0.0

def _merge_junction(raw):
    return 0.55 if raw.get("junction_control") == 1 else 0.0

def _curve(raw):
    return 0.65 if raw.get("road_character_encoded") == 1 else 0.0

def _urban_traffic(raw):
    if raw.get("is_urban") == 1 and raw.get("vehicles_avg", 0) >= 2:
        return 0.5
    return 0.0

def _poor_surface(raw):
    cond = raw.get("road_condition", 0)
    if cond == 1:
        return 0.6   # potholes
    if cond == 2:
        return 0.5   # other defects
    return 0.0

def _weather(raw):
    return 0.55 if raw.get("weather_risk") == 1 else 0.0

def _hotspot(raw):
    count = raw.get("accident_count_6mo", 0)
    sev   = raw.get("severity_numeric", 0)
    if count >= 15:
        return round(min(0.95, 0.6 + (count - 15) * 0.01 + 0.1 * sev), 2)
    return 0.0


DETECTORS = {
    "highway":        _highway,
    "roundabout":     _roundabout,
    "signal":         _signal,
    "merge_junction": _merge_junction,
    "curve":          _curve,
    "urban_traffic":  _urban_traffic,
    "poor_surface":   _poor_surface,
    "weather":        _weather,
    "hotspot":        _hotspot,
}


# ── Phrasing bank — a few natural variants per factor ──────────────────────
# {count} gets substituted where relevant (currently only hotspot).
TEMPLATES = {
    "highway": [
        ("High-speed corridor",
         "Vehicles typically travel at higher speeds along this stretch, cutting down reaction time.",
         "Keep extra distance and avoid sudden lane changes."),
        ("Fast-moving traffic ahead",
         "This is a high-speed section where stopping distances are longer.",
         "Match the flow of traffic and stay alert for merging vehicles."),
    ],
    "roundabout": [
        ("Multi-entry roundabout",
         "Traffic merges in from several directions at once here.",
         "Yield to vehicles already in the roundabout before entering."),
        ("Busy roundabout",
         "This roundabout sees frequent crossing traffic from multiple approaches.",
         "Slow down and check all directions before proceeding."),
    ],
    "signal": [
        ("Traffic signal zone",
         "Expect frequent stopping and starting near this signal.",
         "Be ready to brake — sudden stops are common here."),
        ("Signal-controlled junction",
         "Vehicles queue and release in bursts at this junction.",
         "Watch for vehicles accelerating quickly after the signal changes."),
    ],
    "merge_junction": [
        ("Junction ahead",
         "Vehicles change direction or merge at this point.",
         "Maintain a safe following distance and signal your intentions early."),
        ("Merge point",
         "Traffic converges from another direction here.",
         "Check your mirrors and be ready to adjust speed."),
    ],
    "curve": [
        ("Sharp road curve",
         "The road bends here, which can reduce how far ahead you can see.",
         "Slow down before entering the curve, not during it."),
        ("Winding stretch",
         "Limited visibility around this bend increases the reaction time needed.",
         "Reduce speed and stay in your lane through the curve."),
    ],
    "urban_traffic": [
        ("Dense urban traffic",
         "This area sees heavy, stop-and-go city traffic.",
         "Stay alert for pedestrians and vehicles stopping suddenly."),
        ("Congested city stretch",
         "Traffic density here is higher than the surrounding route.",
         "Keep a safe gap — sudden braking is common in dense traffic."),
    ],
    "poor_surface": [
        ("Poor road surface",
         "This section has known surface defects such as potholes.",
         "Reduce speed to avoid sudden swerving or tyre damage."),
        ("Uneven road ahead",
         "Road condition here is rougher than average.",
         "Drive cautiously and avoid hard braking on the uneven surface."),
    ],
    "weather": [
        ("Weather-sensitive stretch",
         "This section is more affected by rain or reduced visibility.",
         "Slow down and increase following distance in poor weather."),
        ("Rain-prone area",
         "Wet conditions here can reduce tyre grip significantly.",
         "Avoid sudden braking or sharp turns if the road is wet."),
    ],
    "hotspot": [
        ("Historical accident hotspot",
         "{count} accidents have been recorded near this location in the last 6 months.",
         "Stay extra alert — this location has a track record of incidents."),
        ("Known risk zone",
         "This stretch has recorded {count} accidents recently, higher than nearby areas.",
         "Drive defensively and reduce speed through this zone."),
    ],
}


# ── Selection: pick the strongest factor, avoid back-to-back repeats ───────
def _score_factors(raw):
    scores = {}
    for key, fn in DETECTORS.items():
        c = fn(raw)
        if c > 0:
            scores[key] = c
    return scores


def select_factor(raw, recent_factors=None):
    """
    recent_factors: list of factor keys used by the last few markers on
    this route, most recent last. Used to avoid the same factor firing
    on consecutive markers when a close runner-up exists.
    """
    recent_factors = recent_factors or []
    scores = _score_factors(raw)
    if not scores:
        return None, 0.0

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_key, top_conf = ranked[0]

    if recent_factors and recent_factors[-1] == top_key and len(ranked) > 1:
        second_key, second_conf = ranked[1]
        if second_conf >= top_conf - 0.25:   # close enough to swap in
            return second_key, second_conf

    return top_key, top_conf


def explain_segment(raw, risk_level, recent_factors=None):
    factor, confidence = select_factor(raw, recent_factors)

    if factor is None:
        return {
            "title": "Road conditions",
            "description": "No significant road-related concerns were detected here.",
            "advice": "Continue following safe driving practices.",
            "factor": None,
            "confidence": 0.0,
        }

    variants = TEMPLATES[factor]
    idx = _variant_index(len(variants), raw.get("lat", 0), raw.get("lng", 0), factor)
    title, desc_t, advice_t = variants[idx]

    evidence = {"count": raw.get("accident_count_6mo", 0)}
    return {
        "title": title,
        "description": desc_t.format(**evidence),
        "advice": advice_t.format(**evidence),
        "factor": factor,
        "confidence": round(confidence, 2),
    }


def is_important_segment(raw, risk_level, insight=None):
    """
    Decide whether this segment deserves an AI Insight marker.
    Now based on actual detector confidence instead of a feature
    that fires on almost every segment.
    """
    if risk_level == 2:
        return True
    if insight and insight.get("confidence", 0) >= 0.6:
        return True
    return False


if __name__ == "__main__":
    sample_raw = {
        "road_type_encoded": 4,
        "junction_control": 1,
        "road_character_encoded": 0,
        "weather_risk": 0,
        "road_condition": 0,
        "is_urban": 1,
        "vehicles_avg": 2,
        "accident_count_6mo": 20,
        "severity_numeric": 2,
        "lat": 12.97, "lng": 77.59,
    }
    insight = explain_segment(sample_raw, 2)
    print(insight)
    print("important:", is_important_segment(sample_raw, 2, insight))