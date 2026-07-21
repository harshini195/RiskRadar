// ── RouteCard ─────────────────────────────────────────────────────────────

import React from 'react';

export default function RouteCard({ route, active, onClick }) {
  const score = Number(route.risk_score) || 0;

  const normalizedScore =
    score > 1 ? score / 100 : score;

  const color = normalizedScore >= 0.75
    ? '#ef4444'
    : normalizedScore >= 0.45
    ? '#f59e0b'
    : '#22c55e';

  const badgeClass = normalizedScore >= 0.75
    ? 'badge-high'
    : normalizedScore >= 0.45
    ? 'badge-mod'
    : 'badge-safe';

  const badgeLabel = normalizedScore >= 0.75
    ? 'HIGH RISK'
    : normalizedScore >= 0.45
    ? 'MODERATE'
    : route.recommended
    ? 'SAFEST'
    : 'LOW RISK';

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
        <span>⚠ {((route.risk_score > 1    ? route.risk_score / 100    : route.risk_score) * 100).toFixed(0)}%</span>
      </div>
      <div className="route-bar-bg">
        <div
          className="route-bar-fill"
          style={{ width: (normalizedScore * 100) + '%', background: color }}
        />
      </div>

      {route.hotspot_count > 0 && (
        <div
          className="route-hotspot-tag"
          title={route.hotspots_on_route.map(h => h.name).join(', ')}
          style={{
            marginTop: 6,
            fontSize: 12,
            fontWeight: 600,
            color: route.has_high_risk_hotspot ? '#ef4444' : '#f59e0b',
          }}
        >
          🚧 {route.hotspot_count} known hotspot{route.hotspot_count > 1 ? 's' : ''} on this route
          {route.has_high_risk_hotspot ? ' (incl. high-risk)' : ''}
        </div>
      )}

      {route.recommended && (
        <div className="recommended-tag">✓ Recommended — lowest accident risk</div>
      )}
    </div>
  );
}