import React, { useState, useEffect, useRef } from 'react';
import RouteCard from './RouteCard';
import { getModelMetrics } from '../utils/api';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
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
            {t === 'routes' ? 'Routes' : t === 'alerts' ? 'Risk Alerts' : 'ML Model'}
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
function riskColor(risk) {
  return risk >= 67 ? '#ef4444' : risk >= 34 ? '#f59e0b' : '#22c55e';
}

function riskLabel(risk) {
  return risk >= 67 ? 'High' : risk >= 34 ? 'Moderate' : 'Low';
}

function RiskCard({ point }) {
  const color = riskColor(point.risk);
  return (
    <div
      style={{
        background: "#1f2937",
        border: `1px solid ${color}55`,
        borderLeft: `4px solid ${color}`,
        borderRadius: 10,
        padding: 12,
        marginBottom: 10,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontWeight: 700 }}>📍 {point.distance} km</span>
        <span style={{ color, fontWeight: 700 }}>{point.risk}%</span>
      </div>

      {point.title && (
        <div style={{ marginTop: 8, fontWeight: 700 }}>
          {point.title}
        </div>
      )}

      {point.description && (
        <div style={{ marginTop: 4, color: "#cbd5e1", fontSize: 13 }}>
          {point.description}
        </div>
      )}

      {point.advice && (
        <div style={{ marginTop: 8, color: "#4ade80", fontSize: 13, fontWeight: 600 }}>
          💡 {point.advice}
        </div>
      )}
    </div>
  );
}

function AlertsPanel({ selectedRoute }) {
  const [selectedDistance, setSelectedDistance] = useState(null);

  // Reset the selection whenever the user picks a different route.
  useEffect(() => {
    setSelectedDistance(null);
  }, [selectedRoute]);

  if (!selectedRoute)
    return (
      <div className="empty-msg">
        Select a route first.
      </div>
    );

  const trend = selectedRoute.risk_trend || [];
  const sorted = [...trend].sort((a, b) => a.distance - b.distance);
  const cards = selectedDistance == null
    ? sorted
    : sorted.filter(p => p.distance === selectedDistance);

  const handleSelect = (distance) => {
    setSelectedDistance(prev => (prev === distance ? null : distance));
  };

  return (
    <div>
      <h3 style={{ marginBottom: 12 }}>
        Risk Trend Along Route
      </h3>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={sorted}>

          <CartesianGrid stroke="#ffffff" strokeOpacity={0.15} strokeDasharray="3 3" />

          <XAxis
            dataKey="distance"
            unit=" km"
            stroke="#ffffff"
            tick={{ fill: "#ffffff" }}
            axisLine={{ stroke: "#ffffff" }}
            tickLine={{ stroke: "#ffffff" }}
          />

          <YAxis
            domain={[0, 100]}
            unit="%"
            stroke="#ffffff"
            tick={{ fill: "#ffffff" }}
            axisLine={{ stroke: "#ffffff" }}
            tickLine={{ stroke: "#ffffff" }}
          />

          <Line
            type="monotone"
            dataKey="risk"
            stroke="#cbeff5d8"
            strokeWidth={3}
            isAnimationActive={false}
            dot={(props) => {
              const { cx, cy, payload, index } = props;
              const isSelected = selectedDistance === payload.distance;
              return (
                <circle
                  key={`dot-${index}`}
                  cx={cx}
                  cy={cy}
                  r={isSelected ? 6 : 3}
                  fill={riskColor(payload.risk)}
                  stroke={isSelected ? "#ffffff" : "none"}
                  strokeWidth={isSelected ? 2 : 0}
                  style={{ cursor: "pointer" }}
                  onClick={() => handleSelect(payload.distance)}
                />
              );
            }}
            activeDot={(props) => {
              const { cx, cy, payload, index } = props;
              const isSelected = selectedDistance === payload.distance;
              return (
                <circle
                  key={`active-dot-${index}`}
                  cx={cx}
                  cy={cy}
                  r={isSelected ? 6 : 4}
                  fill={riskColor(payload.risk)}
                  stroke="#ffffff"
                  strokeWidth={2}
                  style={{ cursor: "pointer" }}
                  onClick={() => handleSelect(payload.distance)}
                />
              );
            }}
          />

        </LineChart>
      </ResponsiveContainer>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", margin: "14px 0 10px" }}>
        <span style={{ fontWeight: 700, fontSize: 13, color: "#94a3b8" }}>
          {selectedDistance == null
            ? `All points (${sorted.length})`
            : `Showing point at ${selectedDistance} km`}
        </span>
        {selectedDistance != null && (
          <button
            onClick={() => setSelectedDistance(null)}
            style={{
              background: "transparent",
              border: "1px solid #374151",
              color: "#94a3b8",
              borderRadius: 6,
              padding: "4px 10px",
              fontSize: 12,
              cursor: "pointer",
            }}
          >
            Show all
          </button>
        )}
      </div>

      <div>
        {cards.length === 0 ? (
          <div className="empty-msg">No risk points on this route.</div>
        ) : (
          cards.map((p, i) => <RiskCard key={`${p.distance}-${i}`} point={p} />)
        )}
      </div>
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
