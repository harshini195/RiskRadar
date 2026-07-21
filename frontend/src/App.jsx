import React, { useState, useCallback } from 'react';
import MapView from './components/MapView';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import { analyzeRoutes } from './utils/api';
import './App.css';

export default function App() {
  const [routes, setRoutes]               = useState([]);
  const [selectedRoute, setSelectedRoute] = useState(null);
  const [hotspots, setHotspots]           = useState([]);
  const [loading, setLoading]             = useState(false);
  const [error, setError]                 = useState(null);
  const [origin, setOrigin]               = useState('Hebbal, Bangalore');
  const [destination, setDestination]     = useState('Varthur, Bangalore');
  const [alert, setAlert]                 = useState(null);

  const handleAnalyze = useCallback(async () => {
    if (!origin || !destination) return;
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeRoutes(origin, destination);
      setRoutes(data.routes);
      setHotspots(data.hotspots || []);
      // Auto-select recommended (safest) route
      const rec = data.routes.find(r => r.recommended) || data.routes[0];
      setSelectedRoute(rec);
      if (rec?.has_high_risk_hotspot) {
        const names = rec.hotspots_on_route
          .filter(h => h.risk_score >= 0.7)
          .map(h => h.name)
          .join(', ');
        triggerAlert(`⚠ Known accident hotspot on your route: ${names}. Drive cautiously.`);
      } else if (rec?.risk_score >= 0.7) {
        triggerAlert('⚠ High-risk route detected. A safer alternative is available.');
      } else if (rec?.hotspot_count > 0) {
        triggerAlert(`⚠ ${rec.hotspot_count} accident hotspot(s) along your route.`);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [origin, destination]);

  const triggerAlert = (msg) => {
    setAlert(msg);
    // Web Speech API voice alert — speak the actual message (route-specific:
    // names the hotspot, states the count, etc.) instead of a generic line,
    // so it's not identical every time regardless of what actually happened.
    if (window.speechSynthesis) {
      // Strip the leading ⚠ emoji — speech synthesis either skips it
      // silently or reads it awkwardly as "warning sign".
      const spokenText = msg.replace(/^⚠\s*/, '');
      window.speechSynthesis.cancel(); // don't queue/stack overlapping alerts
      const u = new SpeechSynthesisUtterance(spokenText);
      u.rate = 0.9;
      window.speechSynthesis.speak(u);
    }
    setTimeout(() => setAlert(null), 5000);
  };

  return (
    <div className="app-root">
      <Header routes={routes} hotspots={hotspots} />

      <div className="app-body">
        <Sidebar
          origin={origin}
          destination={destination}
          onOriginChange={setOrigin}
          onDestinationChange={setDestination}
          onAnalyze={handleAnalyze}
          routes={routes}
          selectedRoute={selectedRoute}
          onSelectRoute={setSelectedRoute}
          loading={loading}
          error={error}
        />

        <MapView
          routes={routes}
          selectedRoute={selectedRoute}
          hotspots={hotspots}
          origin={origin}
          destination={destination}
          analyzed={routes.length > 0}
          onHotspotsUpdate={setHotspots}
          triggerAlert={triggerAlert}
        />
      </div>

      {alert && (
        <div className="alert-banner" role="alert">
          {alert}
        </div>
      )}
    </div>
  );
}

