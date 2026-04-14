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
      if (rec?.risk_score >= 0.7) {
        triggerAlert('⚠ High-risk route detected. A safer alternative is available.');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [origin, destination]);

  const triggerAlert = (msg) => {
    setAlert(msg);
    // Web Speech API voice alert
    if (window.speechSynthesis) {
      const u = new SpeechSynthesisUtterance(
        'Warning: Accident prone zone ahead. Drive cautiously.'
      );
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
