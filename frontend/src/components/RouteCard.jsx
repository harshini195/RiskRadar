// ── RouteCard ─────────────────────────────────────────────────────────────

import React from 'react';

export default function RouteCard({ route, active, onClick }) {
  const color = route.risk_score >= 0.7
    ? '#ef4444' : route.risk_score >= 0.4
    ? '#f59e0b' : '#22c55e';

  const badgeClass = route.risk_score >= 0.7
    ? 'badge-high' : route.risk_score >= 0.4
    ? 'badge-mod' : 'badge-safe';

  const badgeLabel = route.risk_score >= 0.7
    ? 'HIGH RISK' : route.risk_score >= 0.4
    ? 'MODERATE' : route.recommended ? 'SAFEST' : 'LOW RISK';

  return (
    <div
      className={`route-card ${active ? 'active' : ''} ${route.recommended ? 'recommended' : ''}`}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={e => e.key === 'Enter' && onClick()}
    >
      <div className="route-card-head">
        <span className="route-name" style={{ color }}>{route.summary}</span>
        <span className={`badge ${badgeClass}`}>{badgeLabel}</span>
      </div>
      <div className="route-meta">
        <span>📍 {route.distance_km} km</span>
        <span>⏱ {route.duration_min} min</span>
        <span>⚠ {(route.risk_score * 100).toFixed(0)}% risk</span>
      </div>
      <div className="route-bar-bg">
        <div
          className="route-bar-fill"
          style={{ width: (route.risk_score * 100) + '%', background: color }}
        />
      </div>
      {route.recommended && (
        <div className="recommended-tag">✓ Recommended — lowest accident risk</div>
      )}
    </div>
  );
}
