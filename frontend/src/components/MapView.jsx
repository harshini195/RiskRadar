import React, { useEffect, useRef, useState } from 'react';
import Papa from 'papaparse';

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
}) {
  const mapRef    = useRef(null);
  const mapObj    = useRef(null);
  const polylines = useRef([]);
  const markers   = useRef([]);
  const csvMarkers = useRef([]);
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
  useEffect(() => {
    if (!mapObj.current || !csvLoaded) return;

    // Clear old CSV markers
    csvMarkers.current.forEach(m => m.setMap(null));
    csvMarkers.current = [];

    if (!showCSV) return;

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
  }, [csvData, csvLoaded, showCSV, filterLevel]);

  // ── Draw API hotspot markers ──────────────────────────────
  useEffect(() => {
    if (!mapObj.current) return;
    markers.current.forEach(m => m.setMap(null));
    markers.current = [];
    if (!showHotspots) return;

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
  }, [hotspots, showHotspots]);

  // ── Draw route polylines ──────────────────────────────────
  useEffect(() => {
    if (!mapObj.current || !window.google?.maps?.geometry) return;
    polylines.current.forEach(p => p.setMap(null));
    polylines.current = [];

    routes.forEach((route, i) => {
      // Debug: log risk_score
      console.log('Route', route.summary || route.route_index, 'risk_score:', route.risk_score);
      let color;
      if (route.risk_score >= 0.7) {
        color = '#ef4444';
      } else if (route.risk_score >= 0.5) {
        color = '#f59e0b';
      } else {
        color = '#22c55e';
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
              Risk: ${route.risk_label} (${(route.risk_score * 100).toFixed(0)}%)
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