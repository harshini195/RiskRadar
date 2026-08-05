import React, { useState, useEffect, useRef } from 'react';
import RouteCard from './RouteCard';
import { getModelMetrics } from '../utils/api';
export default function Sidebar({
  origin, destination, onOriginChange, onDestinationChange,
  onAnalyze, routes, selectedRoute, onSelectRoute, loading, error,
}) {
  const [tab, setTab] = useState('routes');
  const originRef = useRef(null);
  const destRef = useRef(null);
  useEffect(() => {
    if (!window.google?.maps?.places?.Autocomplete) return;

    const originAuto = new window.google.maps.places.Autocomplete(originRef.current);
    const destAuto = new window.google.maps.places.Autocomplete(destRef.current);

    originAuto.addListener("place_changed", () => {
      const place = originAuto.getPlace();
      if (place.formatted_address) {
        onOriginChange(place.formatted_address);
      }
    });

    destAuto.addListener("place_changed", () => {
      const place = destAuto.getPlace();
      if (place.formatted_address) {
        onDestinationChange(place.formatted_address);
      }
    });
  }, []);

  return (
    <aside className="sidebar">
      {/* Route planner */}
      <div className="sidebar-section">
        <div className="sec-title">Route Planner</div>
        <div className="route-inputs">
          <div className="input-row">
            <div className="dot dot-start" />
            <input
              ref={originRef}
              value={origin}
              onChange={e => onOriginChange(e.target.value)}
              placeholder="Starting location..."
            />
          </div>
          <div className="input-connector" />
          <div className="input-row">
            <div className="dot dot-end" />
            <input
              ref={destRef}
              value={destination}
              onChange={e => onDestinationChange(e.target.value)}
              placeholder="Destination..."
            />
          </div>
        </div>
        <button
          className="btn-analyze"
          onClick={onAnalyze}
          disabled={loading}
        >
          {loading ? '⏳ Analyzing...' : '🔍 Analyze Routes'}
        </button>
        {error && <div className="error-msg">⚠ {error}</div>}
      </div>

      {/* Tabs */}
      <div className="tabs">
        {['routes', 'alerts', 'model'].map(t => (
          <button
            key={t}
            className={`tab ${tab === t ? 'active' : ''}`}
            onClick={() => setTab(t)}
          >
            {t === 'routes' ? 'Routes' : t === 'alerts' ? 'Alerts' : 'ML Model'}
          </button>
        ))}
      </div>

      <div className="tab-body">
        {tab === 'routes' && (
          <div>
            {routes.length === 0 ? (
              <p className="empty-msg">
                Analyze a route to see safety-aware alternatives ranked by risk.
              </p>
            ) : (
              routes.map((r, i) => (
                <RouteCard
                  key={i}
                  route={r}
                  active={selectedRoute?.route_index === r.route_index}
                  onClick={() => onSelectRoute(r)}
                />
              ))
            )}
          </div>
        )}

        {tab === 'alerts' && (
    <AlertsPanel selectedRoute={selectedRoute} />
)}
        {tab === 'model' && <MLPanel />}
      </div>
    </aside>
  );
}
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

function RiskTooltip({ active, payload }) {

  if (!active || !payload || !payload.length) return null;

  const point = payload[0].payload;

  return (
    <div
      style={{
        background: "#1f2937",
        color: "white",
        padding: "12px",
        borderRadius: "10px",
        border: "1px solid #374151",
        maxWidth: "220px",
      }}
    >
      <div style={{ fontWeight: "bold" }}>
        📍 {point.distance} km
      </div>

      <div style={{ marginTop: 8 }}>
        Risk: <b>{point.risk}%</b>
      </div>

      {point.title && (
        <div style={{ marginTop: 10 }}>
          <b>{point.title}</b>
        </div>
      )}

      {point.description && (
        <div style={{ marginTop: 4 }}>
          {point.description}
        </div>
      )}

      {point.advice && (
        <div style={{ marginTop: 8, color: "#4ade80" }}>
          💡 {point.advice}
        </div>
      )}

    </div>
  );
}

function AlertsPanel({ selectedRoute }) {
  if (!selectedRoute)
    return (
      <div className="empty-msg">
        Select a route first.
      </div>
    );

  return (
    <div style={{ height: 320 }}>

      <h3 style={{ marginBottom: 12 }}>
        Risk Trend Along Route
      </h3>

      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={selectedRoute.risk_trend}>

          <CartesianGrid strokeDasharray="3 3" />

          <XAxis
            dataKey="distance"
            unit=" km"
          />

          <YAxis
            domain={[0,100]}
          />

          <Tooltip content={<RiskTooltip />} />

          <Line
            type="monotone"
            dataKey="risk"
            stroke="#ef4444"
            strokeWidth={3}
            dot
          />

        </LineChart>
      </ResponsiveContainer>

    </div>
  );
}
function MLPanel() {
  const [metrics, setMetrics] = useState(null);
  const [features, setFeatures] = useState(null);
  const [bestModel, setBestModel] = useState(null);
  const [testSamples, setTestSamples] = useState(null);
  const [split, setSplit] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    getModelMetrics()
      .then(data => {
        setMetrics(data.metrics);
        setFeatures(data.features);
        setBestModel(data.best_model);
        setTestSamples(data.test_samples);
        setSplit(data.split);
        setLoading(false);
      })
      .catch(e => {
        setError(e.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <div className="ml-panel">Loading model metrics...</div>;
  if (error) return <div className="ml-panel error-msg">{error}</div>;
  if (!metrics || !features) return null;

  return (
    <div className="ml-panel">
      <div className="ml-grid">
        {metrics.map(m => (
          <div key={m.label} className="ml-card">
            <div className="ml-label">{m.label}</div>
            <div className="ml-value" style={{ color: m.color }}>{m.value}</div>
            <div className="ml-bar-track">
              <div className="ml-bar-fill" style={{ width: m.pct + '%', background: m.color }} />
            </div>
          </div>
        ))}
      </div>
      <div className="sec-title" style={{ marginTop: 14, marginBottom: 8 }}>Feature Importance</div>
      {features.map(f => (
        <div key={f.name} className="feat-row">
          <div className="feat-header">
            <span>{f.name}</span>
            <span className="feat-pct">{f.pct}%</span>
          </div>
          <div className="ml-bar-track">
            <div className="ml-bar-fill" style={{ width: f.pct + '%' }} />
          </div>
        </div>
      ))}
      <div className="model-note">
        {bestModel || 'Model'}
        {testSamples ? ` · ${testSamples.toLocaleString()} test samples` : ''}
        {split ? ` · ${split}` : ''}
      </div>
    </div>
  );
}
