import React, { useEffect, useRef, useState } from 'react';
import Papa from 'papaparse';
import { getHotspots, getHotspotsOnRoute } from '../utils/api';

const RISK_COLORS = { Low: '#22c55e', Moderate: '#f59e0b', High: '#ef4444' };

const RISK_LEVEL_MAP = { 0: 'Low', 1: 'Moderate', 2: 'High' };
const RISK_SCORE_MAP = { 0: 0.2,   1: 0.55,       2: 0.85  };

const CAUSE_MAP = {
  0: 'Overspeeding',
  1: 'Signal Jumping',
  2: 'Reckless Driving',
  3: 'Wrong Side',
  4: 'Drunk Driving',
};

function riskColor(level) {
  if (level === 2) return '#ef4444';
  if (level === 1) return '#f59e0b';
  return '#22c55e';
}

export default function MapView({
  routes, selectedRoute, hotspots,
  origin, destination, analyzed,
  onHotspotsUpdate, triggerAlert,
}) {
  const mapRef    = useRef(null);
  const mapObj    = useRef(null);
  const polylines = useRef([]);
  const markers   = useRef([]);
  const csvMarkers = useRef([]);
  const routeHotspotMarkers = useRef([]);
  const aiInsightMarkers = useRef([]);
  const infoWin   = useRef(null);

  const [legend,        setLegend]        = useState(true);
  const [showHotspots,  setShowHotspots]  = useState(true);
  const [showCSV,       setShowCSV]       = useState(true);
  const [csvData,       setCsvData]       = useState([]);
  const [csvLoaded,     setCsvLoaded]     = useState(false);
  const [filterLevel,   setFilterLevel]   = useState('All'); // All | Low | Moderate | High
  const [csvError,      setCsvError]      = useState(null);

  // ── Init map ──────────────────────────────────────────────
  useEffect(() => {
    const initMap = () => {
      if (!window.google?.maps?.Map || mapObj.current) return;
      mapObj.current = new window.google.maps.Map(mapRef.current, {
        center: { lat: 15.3173, lng: 75.7139 }, // centre of Karnataka
        zoom: 7,
        styles: DARK_STYLE,
      });
      infoWin.current = new window.google.maps.InfoWindow();
    };

    if (window.google?.maps?.Map) { initMap(); return; }
    const iv = setInterval(() => {
      if (window.google?.maps?.Map) { clearInterval(iv); initMap(); }
    }, 100);
    return () => clearInterval(iv);
  }, []);

  // ── Load CSV from /public folder ──────────────────────────
  useEffect(() => {
    fetch('ml/data/cleaned_accidents.csv')          // 👈 put your CSV in /public
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then(text => {
        const result = Papa.parse(text, {
          header: true,
          skipEmptyLines: true,
          dynamicTyping: true,
        });
        // Keep only rows with valid coordinates
        const clean = result.data.filter(
          r => r.Latitude && r.Longitude &&
               !isNaN(r.Latitude) && !isNaN(r.Longitude)
        );
        setCsvData(clean);
        setCsvLoaded(true);
      })
      .catch(err => {
        console.error('CSV load error:', err);
        setCsvError('Could not load accident dataset.');
      });
  }, []);

  // ── Draw CSV markers ──────────────────────────────────────
  // Raw historical accident records from the bundled CSV — shown only
  // before a route has been analyzed (general area awareness), same as
  // the API hotspot circles. These plot wherever the record's lat/lon
  // happens to be, with no relation to any specific route, so once a
  // route exists we defer to the route-matched warning triangles instead.
  useEffect(() => {
    if (!mapObj.current || !csvLoaded) return;

    // Clear old CSV markers
    csvMarkers.current.forEach(m => m.setMap(null));
    csvMarkers.current = [];

    if (!showCSV || analyzed) return;

    const filtered = filterLevel === 'All'
      ? csvData
      : csvData.filter(r => RISK_LEVEL_MAP[r.risk_level] === filterLevel);

    filtered.forEach(row => {
      const level  = row.risk_level ?? 0;
      const color  = riskColor(level);
      const label  = RISK_LEVEL_MAP[level] ?? 'Unknown';
      const score  = RISK_SCORE_MAP[level] ?? 0.2;
      const cause  = CAUSE_MAP[row.main_cause_encoded] ?? 'Unknown';

      const marker = new window.google.maps.Marker({
        position: { lat: row.Latitude, lng: row.Longitude },
        map: mapObj.current,
        icon: {
          path: window.google.maps.SymbolPath.CIRCLE,
          scale: 5 + score * 10,           // size by risk
          fillColor: color,
          fillOpacity: 0.75,
          strokeColor: '#ffffff',
          strokeWeight: 1,
        },
        title: row.DISTRICTNAME,
      });

      marker.addListener('click', () => {
        infoWin.current.setContent(`
          <div style="font-family:sans-serif;min-width:180px;padding:4px">
            <b style="font-size:13px">${row.DISTRICTNAME}</b>
            <span style="color:#888;font-size:11px"> · ${row.Year}</span><br/>
            <span style="color:${color};font-weight:700;font-size:13px">
              Risk: ${label}
            </span><br/>
            <span style="color:#555;font-size:12px">
              Accidents (6 mo): ${row.accident_count_6mo === 500 ? '500+' : row.accident_count_6mo}<br/>
              Main cause: ${cause}<br/>
              Severity: ${row.severity_numeric}/3<br/>
              Urban: ${row.is_urban ? 'Yes' : 'No'} · 
              Highway: ${row.is_highway ? 'Yes' : 'No'}
            </span>
          </div>
        `);
        infoWin.current.open(mapObj.current, marker);
      });

      csvMarkers.current.push(marker);
    });
  }, [csvData, csvLoaded, showCSV, filterLevel, analyzed]);

  // ── Draw API hotspot markers ──────────────────────────────
  // Only shown before a route has been analyzed (general area awareness).
  // Once a route exists, the "hotspots ON the selected route" effect below
  // takes over — showing only genuinely relevant, road-snapped markers
  // instead of every hotspot in the whole search radius.
  useEffect(() => {
    if (!mapObj.current) return;
    markers.current.forEach(m => m.setMap(null));
    markers.current = [];
    if (!showHotspots || analyzed) return;

    hotspots.forEach(h => {
      const color = h.risk_score >= 0.7 ? '#ef4444'
                  : h.risk_score >= 0.4 ? '#f59e0b' : '#22c55e';
      const marker = new window.google.maps.Marker({
        position: { lat: h.lat, lng: h.lon },
        map: mapObj.current,
        icon: {
          path: window.google.maps.SymbolPath.CIRCLE,
          scale: 10 + h.risk_score * 14,
          fillColor: color,
          fillOpacity: 0.75,
          strokeColor: '#ffffff',
          strokeWeight: 1.5,
        },
        title: h.name,
      });
      marker.addListener('click', () => {
        infoWin.current.setContent(`
          <div style="font-family:sans-serif;min-width:160px">
            <b style="font-size:13px">${h.name}</b><br/>
            <span style="color:${color};font-weight:600">
              Risk: ${(h.risk_score * 100).toFixed(0)}%
            </span><br/>
            <span style="color:#666;font-size:12px">
              ${h.accidents} accidents (6 mo)<br/>
              Main cause: ${h.main_cause}
            </span>
          </div>
        `);
        infoWin.current.open(mapObj.current, marker);
      });
      markers.current.push(marker);
    });
  }, [hotspots, showHotspots, analyzed]);

  // ── Draw hotspots ON the selected route (distinct warning markers) ──
  useEffect(() => {
    if (!mapObj.current) return;
    routeHotspotMarkers.current.forEach(m => m.setMap(null));
    routeHotspotMarkers.current = [];

    // This is the layer the "Show/Hide Route Hotspots" button actually
    // controls once a route is analyzed (the other two hotspot layers
    // are intentionally hidden at that point — see their own effects).
    if (!showHotspots) return;

    const onRoute = selectedRoute?.hotspots_on_route || [];
    onRoute.forEach(h => {
      const color = h.risk_score >= 0.7 ? '#ef4444'
                  : h.risk_score >= 0.4 ? '#f59e0b' : '#eab308';

      // Use the snapped point ON the route (route_lat/route_lon) for the
      // marker's visual position, not the hotspot's raw lat/lon — the real
      // recorded location can be up to HOTSPOT_BUFFER_KM off the actual
      // road, which otherwise makes the marker look disconnected from the
      // route line even though it correctly matched.
      const markerLat = h.route_lat ?? h.lat;
      const markerLng = h.route_lon ?? h.lon;

      const marker = new window.google.maps.Marker({
        position: { lat: markerLat, lng: markerLng },
        map: mapObj.current,
        zIndex: 999,
        icon: {
          path: 'M12 2 L22 20 L2 20 Z',   // warning triangle
          fillColor: color,
          fillOpacity: 0.95,
          strokeColor: '#ffffff',
          strokeWeight: 1.5,
          scale: 1.4,
          anchor: new window.google.maps.Point(12, 20),
        },
        title: `⚠ ${h.name} — ${h.distance_from_start_km} km into route`,
      });

      marker.addListener('click', () => {
        infoWin.current.setContent(`
          <div style="font-family:sans-serif;min-width:180px">
            <b style="font-size:13px">⚠ ${h.name}</b><br/>
            <span style="color:${color};font-weight:600">
              Risk: ${(h.risk_score * 100).toFixed(0)}%
            </span><br/>
            <span style="color:#666;font-size:12px">
              ${h.distance_from_start_km} km into this route
              (${Math.round(h.distance_to_route_km * 1000)} m off road)<br/>
              ${h.accidents} accidents (6 mo) · ${h.main_cause}
            </span>
          </div>
        `);
        infoWin.current.open(mapObj.current, marker);
      });

      routeHotspotMarkers.current.push(marker);
    });
  }, [selectedRoute, showHotspots]);

  // ── Poll for live hotspot updates every 45s ────────────────────────
  // Reuses the same backend logic (getHotspots / getHotspotsOnRoute) as
  // the initial route analysis, so "is this hotspot on my route" is
  // decided in exactly one place (utils/geo.py's segment-based match),
  // not duplicated with a weaker check here.
  const knownHotspotIds = useRef(new Set());

  useEffect(() => {
    if (!analyzed) return; // nothing to poll for until a route exists

    let cancelled = false;

    const poll = async () => {
      try {
        const data = await getHotspots(12.97, 77.59, 20);
        if (cancelled) return;

        const freshIds = new Set(data.hotspots.map(h => h.id));
        const isFirstPoll = knownHotspotIds.current.size === 0;
        const newIds = [...freshIds].filter(id => !knownHotspotIds.current.has(id));

        const changed =
          freshIds.size !== knownHotspotIds.current.size ||
          [...freshIds].some(id => !knownHotspotIds.current.has(id));

        if (changed) {
          knownHotspotIds.current = freshIds;
          onHotspotsUpdate?.(data.hotspots);
        }

        // Only alert for hotspots that are genuinely new (skip the very
        // first poll, which would otherwise flag every existing hotspot).
        if (!isFirstPoll && newIds.length > 0 && selectedRoute?.polyline) {
          const onRoute = await getHotspotsOnRoute(selectedRoute.polyline, 0.3);
          if (cancelled) return;
          const newOnRoute = onRoute.hotspots.filter(h => newIds.includes(h.id));
          if (newOnRoute.length > 0) {
            const names = newOnRoute.map(h => h.name).join(', ');
            triggerAlert?.(`⚠ New accident hotspot on your route: ${names}`);
          }
        }
      } catch (err) {
        console.error('Hotspot polling failed:', err);
      }
    };

    poll(); // run once immediately, then on an interval
    const intervalId = setInterval(poll, 45000);

    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, [analyzed, selectedRoute?.polyline]);

  useEffect(() => {

    if (!mapObj.current) return;

    // Remove old markers
    aiInsightMarkers.current.forEach(m => m.setMap(null));
    aiInsightMarkers.current = [];

    if (!selectedRoute?.route_insights) return;

    selectedRoute.route_insights.forEach(point => {

        const color =
            point.risk >= 80
                ? "#ef4444"
                : point.risk >= 40
                ? "#f59e0b"
                : "#22c55e";

        const marker = new window.google.maps.Marker({

            position: {
                lat: point.lat,
                lng: point.lng
            },

            map: mapObj.current,

            label: {
                text: "⚠",
                color: "white",
                fontWeight: "bold"
            },

            icon: {
                path: window.google.maps.SymbolPath.CIRCLE,
                scale: 12,
                fillColor: color,
                fillOpacity: 1,
                strokeColor: "#fff",
                strokeWeight: 2
            }

        });

        // ✅ Click event
        marker.addListener("click", () => {

            const reasons = (point.reasons || [])
                .map(r => `<li>${r}</li>`)
                .join("");

            infoWin.current.setContent(`
                  <div style="
                      width:210px;
                      padding:8px;
                      font-family:Arial,sans-serif;
                  ">

                      <div style="
                          font-weight:700;
                          color:${color};
                          font-size:15px;
                          margin-bottom:6px;
                      ">
                          ⚠ ${point.title}
                      </div>

                      <div style="
                          color:#444;
                          font-size:13px;
                          line-height:1.4;
                      ">
                          ${point.description}
                      </div>

                      <div style="
                          margin-top:10px;
                          color:#16a34a;
                          font-weight:600;
                          font-size:13px;
                      ">
                          💡 ${point.advice}
                      </div>

                  </div>
                  `);

            infoWin.current.open({
                anchor: marker,
                map: mapObj.current
            });

        });

        aiInsightMarkers.current.push(marker);

    });

}, [selectedRoute]);
  // ── Draw route polylines ──────────────────────────────────
  useEffect(() => {
    if (!mapObj.current || !window.google?.maps?.geometry) return;
    polylines.current.forEach(p => p.setMap(null));
    polylines.current = [];

    routes.forEach((route, i) => {
      // Normalize risk score safely
      const score = Number(route.risk_score) || 0;

      const normalizedScore =
        score > 1 ? score / 100 : score;

      console.log(
        'Route',
        route.summary || route.route_index,
        'risk_score:',
        normalizedScore
      );

      let color = '#22c55e';

      if (normalizedScore >= 0.75) {
        color = '#ef4444'; // High Risk
      }
      else if (normalizedScore >= 0.45) {
        color = '#f59e0b'; // Moderate Risk
      }
      const isSelected = selectedRoute?.route_index === route.route_index;
      const path  = window.google.maps.geometry.encoding.decodePath(route.polyline);
      // Start & End markers
const start = path[0];
const end = path[path.length - 1];

new window.google.maps.Marker({
  position: start,
  map: mapObj.current,
  icon: {
    path: window.google.maps.SymbolPath.CIRCLE,
    scale: 6,
    fillColor: '#22c55e',
    fillOpacity: 1,
    strokeColor: '#fff',
    strokeWeight: 2,
  },
});

new window.google.maps.Marker({
  position: end,
  map: mapObj.current,
  icon: {
    path: window.google.maps.SymbolPath.CIRCLE,
    scale: 6,
    fillColor: '#ef4444',
    fillOpacity: 1,
    strokeColor: '#fff',
    strokeWeight: 2,
  },
});
      const poly = new window.google.maps.Polyline({
        path,
        map: mapObj.current,
        strokeColor:   color,
        strokeOpacity: isSelected ? 0.95 : 0.3,
        strokeWeight:  isSelected ? 6 : 3,
        zIndex:        isSelected ? 10 : 1,
      });

      poly.addListener('click', () => {
        infoWin.current.setContent(`
          <div style="font-family:sans-serif">
            <b>${route.summary}</b><br/>
            <span style="color:${color}">
              Risk: ${route.risk_label} (${(normalizedScore * 100).toFixed(0)}%)
              
            </span><br/>
            <span style="color:#666;font-size:12px">
              ${route.distance_km} km · ${route.duration_min} min
            </span>
            ${route.recommended ? '<br/><b style="color:#22c55e">✓ Recommended</b>' : ''}
          </div>
        `);
        infoWin.current.open(mapObj.current);
        infoWin.current.setPosition(path[Math.floor(path.length / 2)]);
      });

      polylines.current.push(poly);
    });
  }, [routes, selectedRoute]);

  // ── Fit map to selected route ───────────────────────────
  useEffect(() => {
    if (!mapObj.current || !window.google?.maps?.geometry) return;
    let route = selectedRoute || routes[0];
    if (!route || !route.polyline) return;
    const path = window.google.maps.geometry.encoding.decodePath(route.polyline);
    if (!path.length) return;
    const bounds = new window.google.maps.LatLngBounds();
    path.forEach(pt => bounds.extend(pt));
    mapObj.current.fitBounds(bounds);
  }, [routes, selectedRoute]);

  // ── UI ────────────────────────────────────────────────────
  return (
    <div style={{ flex: 1, position: 'relative' }}>
      <div ref={mapRef} style={{ width: '100%', height: '100%' }} />

      {/* Controls bar */}
      <div style={{
        position: 'absolute', top: 12, right: 12, zIndex: 10,
        display: 'flex', flexDirection: 'column', gap: 8,
      }}>
       

        {/* API hotspots toggle */}
        <button
          onClick={() => setShowHotspots(p => !p)}
          style={{ ...btnStyle(showHotspots ? '#ef4444' : '#22c55e'), marginRight: '50px' }}
        >
          {showHotspots ? '🔴 Hide Route Hotspots' : '🟢 Show Route Hotspots'}
        </button>
      </div>
      

      {/* CSV load error */}
      {csvError && (
        <div style={{
          position: 'absolute', bottom: 60, right: 12,
          background: '#7f1d1d', color: '#fca5a5',
          padding: '8px 14px', borderRadius: 8, fontSize: 12,
          zIndex: 10,
        }}>
          ⚠ {csvError}
        </div>
      )}

      {/* CSV loaded count */}
      {csvLoaded && (
        <div style={{
          position: 'absolute', bottom: 12, right: 12,
          background: 'rgba(0,0,0,0.6)', color: '#94a3b8',
          padding: '5px 10px', borderRadius: 6, fontSize: 11,
          zIndex: 10,
        }}>
          {csvData.filter(r =>
            filterLevel === 'All' || RISK_LEVEL_MAP[r.risk_level] === filterLevel
          ).length} zones shown
        </div>
      )}

      {/* Legend */}
      {legend && (
        <div className="map-legend">
          <div className="legend-title">Risk Level</div>
          {Object.entries(RISK_COLORS).map(([label, color]) => (
            <div key={label} className="legend-row">
              <div className="legend-dot" style={{ background: color }} />
              <span>{label}</span>
            </div>
          ))}
          <button className="legend-close" onClick={() => setLegend(false)}>×</button>
        </div>
      )}

      {!analyzed && (
        <div className="map-overlay-tip">
          Enter locations and click "Analyze Routes"
        </div>
      )}
    </div>
  );
}

// ── Helpers ───────────────────────────────────────────────
function btnStyle(bg) {
  return {
    padding: '8px 14px',
    borderRadius: 8,
    border: 'none',
    cursor: 'pointer',
    fontWeight: 600,
    fontSize: 13,
    backgroundColor: bg,
    color: '#fff',
    boxShadow: '0 2px 8px rgba(0,0,0,0.4)',
    transition: 'background-color 0.2s',
    whiteSpace: 'nowrap',
  };
}

const DARK_STYLE = [
  { elementType: 'geometry',            stylers: [{ color: '#1a2235' }] },
  { elementType: 'labels.text.stroke',  stylers: [{ color: '#111827' }] },
  { elementType: 'labels.text.fill',    stylers: [{ color: '#8fa3be' }] },
    { featureType: 'road', elementType: 'geometry', stylers: [{ color: '#2a3a52' }] },
  { featureType: 'road.arterial', elementType: 'geometry', stylers: [{ color: '#3a4f6a' }] },
  { featureType: 'road.highway',  elementType: 'geometry', stylers: [{ color: '#4f6a8a' }] },
  { featureType: 'water', elementType: 'geometry', stylers: [{ color: '#0f1929' }] },
  { featureType: 'poi', stylers: [{ visibility: 'off' }] },
];
  

