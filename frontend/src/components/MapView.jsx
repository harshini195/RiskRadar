import React, { useEffect, useRef, useState } from 'react';

const RISK_COLORS = { Low: '#22c55e', Moderate: '#f59e0b', High: '#ef4444' };

function riskColor(score) {
  if (score >= 0.7) return '#ef4444';
  if (score >= 0.4) return '#f59e0b';
  return '#22c55e';
}

export default function MapView({ routes, selectedRoute, hotspots, origin, destination, analyzed }) {
  const mapRef    = useRef(null);
  const mapObj    = useRef(null);
  const polylines = useRef([]);
  const markers   = useRef([]);
  const infoWin   = useRef(null);
  const [legend, setLegend] = useState(true);

  // Init map
  useEffect(() => {
    if (!window.google || mapObj.current) return;
    mapObj.current = new window.google.maps.Map(mapRef.current, {
      center: { lat: 12.97, lng: 77.59 },
      zoom: 12,
      mapId: 'riskradar',
      disableDefaultUI: false,
      styles: DARK_STYLE,
    });
    infoWin.current = new window.google.maps.InfoWindow();
  }, []);

  // Draw hotspot markers
  useEffect(() => {
    if (!mapObj.current) return;
    markers.current.forEach(m => m.setMap(null));
    markers.current = [];

    hotspots.forEach(h => {
      const color = riskColor(h.risk_score);
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
            <b style="font-size:13px">${h.name}</b><br>
            <span style="color:${color};font-weight:600">Risk: ${(h.risk_score * 100).toFixed(0)}%</span><br>
            <span style="color:#666;font-size:12px">
              ${h.accidents} accidents (6 mo)<br>
              Main cause: ${h.main_cause}
            </span>
          </div>
        `);
        infoWin.current.open(mapObj.current, marker);
      });
      markers.current.push(marker);
    });
  }, [hotspots]);

  // Draw route polylines
  useEffect(() => {
    if (!mapObj.current) return;
    polylines.current.forEach(p => p.setMap(null));
    polylines.current = [];

    routes.forEach((route, i) => {
      const isSelected = selectedRoute?.route_index === i;
      const path = window.google.maps.geometry.encoding.decodePath(route.polyline);
      const color = riskColor(route.risk_score);

      const poly = new window.google.maps.Polyline({
        path,
        map: mapObj.current,
        strokeColor: color,
        strokeOpacity: isSelected ? 0.95 : 0.3,
        strokeWeight: isSelected ? 6 : 3,
        zIndex: isSelected ? 10 : 1,
      });

      poly.addListener('click', () => {
        infoWin.current.setContent(`
          <div style="font-family:sans-serif">
            <b>${route.summary}</b><br>
            <span style="color:${color}">Risk: ${route.risk_label} (${(route.risk_score * 100).toFixed(0)}%)</span><br>
            <span style="color:#666;font-size:12px">${route.distance_km} km · ${route.duration_min} min</span>
            ${route.recommended ? '<br><b style="color:#22c55e">✓ Recommended</b>' : ''}
          </div>
        `);
        infoWin.current.open(mapObj.current);
        infoWin.current.setPosition(path[Math.floor(path.length / 2)]);
      });

      polylines.current.push(poly);
    });
  }, [routes, selectedRoute]);

  return (
    <div style={{ flex: 1, position: 'relative' }}>
      <div ref={mapRef} style={{ width: '100%', height: '100%' }} />

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

// Google Maps dark style
const DARK_STYLE = [
  { elementType: 'geometry', stylers: [{ color: '#1a2235' }] },
  { elementType: 'labels.text.stroke', stylers: [{ color: '#111827' }] },
  { elementType: 'labels.text.fill', stylers: [{ color: '#8fa3be' }] },
  { featureType: 'road', elementType: 'geometry', stylers: [{ color: '#2a3a52' }] },
  { featureType: 'road.arterial', elementType: 'geometry', stylers: [{ color: '#3a4f6a' }] },
  { featureType: 'road.highway', elementType: 'geometry', stylers: [{ color: '#4f6a8a' }] },
  { featureType: 'water', elementType: 'geometry', stylers: [{ color: '#0f1929' }] },
  { featureType: 'poi', stylers: [{ visibility: 'off' }] },
];
