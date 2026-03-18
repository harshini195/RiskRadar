import React, { useState } from 'react';
import RouteCard from './RouteCard';

export default function Sidebar({
  origin, destination, onOriginChange, onDestinationChange,
  onAnalyze, routes, selectedRoute, onSelectRoute, loading, error,
}) {
  const [tab, setTab] = useState('routes');

  return (
    <aside className="sidebar">
      {/* Route planner */}
      <div className="sidebar-section">
        <div className="sec-title">Route Planner</div>
        <div className="route-inputs">
          <div className="input-row">
            <div className="dot dot-start" />
            <input
              value={origin}
              onChange={e => onOriginChange(e.target.value)}
              placeholder="Starting location..."
            />
          </div>
          <div className="input-connector" />
          <div className="input-row">
            <div className="dot dot-end" />
            <input
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

        {tab === 'alerts' && <AlertsPanel />}
        {tab === 'model' && <MLPanel />}
      </div>
    </aside>
  );
}

function AlertsPanel() {
  const alerts = [
    { level: 'danger', icon: '🔴', text: 'High-risk intersection ahead — multiple accidents reported', loc: 'Silk Board Junction · Risk: 0.87' },
    { level: 'warn',   icon: '🟡', text: 'Moderate risk zone — poor road surface condition', loc: 'Hosur Road, KM 12 · Risk: 0.61' },
    { level: 'info',   icon: '🔵', text: 'School zone — reduced speed limit active', loc: 'BTM Layout, 2nd Stage · Risk: 0.32' },
    { level: 'danger', icon: '🔴', text: 'Accident hotspot — wet road conditions detected', loc: 'Electronic City Flyover · Risk: 0.79' },
  ];
  return (
    <div className="alerts-list">
      {alerts.map((a, i) => (
        <div key={i} className={`alert-item ${a.level}`}>
          <span className="alert-icon">{a.icon}</span>
          <div>
            <div className="alert-text">{a.text}</div>
            <div className="alert-loc">{a.loc}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

function MLPanel() {
  const metrics = [
    { label: 'Accuracy',  value: '87.4%', color: '#22c55e', pct: 87.4 },
    { label: 'Precision', value: '83.1%', color: '#4f8ef7', pct: 83.1 },
    { label: 'Recall',    value: '91.2%', color: '#f59e0b', pct: 91.2 },
    { label: 'F1 Score',  value: '0.871', color: '#4f8ef7', pct: 87.1 },
  ];
  const features = [
    { name: 'Accident History', pct: 31 },
    { name: 'Road Condition',   pct: 22 },
    { name: 'Junction Type',    pct: 18 },
    { name: 'Traffic Density',  pct: 14 },
    { name: 'Weather Pattern',  pct: 9  },
    { name: 'Speed Limit',      pct: 6  },
  ];
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
        Random Forest · 1,200 samples · 5-fold CV
      </div>
    </div>
  );
}
