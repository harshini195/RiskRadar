// ── RouteCard ─────────────────────────────────────────────────────────────
// FIXED: previously recalculated risk color/label on the frontend using
// thresholds (0.45 / 0.75) that didn't match the backend's thresholds
// (0.34 / 0.67), and ignored route.risk_label + has_high_risk_hotspot
// entirely. A route scoring 0.37 — already "Moderate" per the backend,
// and potentially "High" if a known hotspot sits on it — was rendering
// green/"SAFEST" purely because of the mismatched frontend cutoff.
//
// Now the backend's risk_label (which already folds in the hotspot
// escalation logic from route_routes.py) is the single source of truth.
// The frontend only maps that label to a color/badge — it never
// re-derives risk level from the raw score.

import React from 'react';

// Single source of truth for label -> visual style. Keys must match
// exactly what the backend's risk_label can be: 'Low' | 'Moderate' | 'High'.
const RISK_STYLES = {
  High: { color: '#ef4444', badgeClass: 'badge-high', badgeLabel: 'HIGH RISK' },
  Moderate: { color: '#f59e0b', badgeClass: 'badge-mod', badgeLabel: 'MODERATE' },
  Low: { color: '#22c55e', badgeClass: 'badge-safe', badgeLabel: 'LOW RISK' },
};

export default function RouteCard({ route, active, onClick }) {
  const score = Number(route.risk_score) || 0;
  const normalizedScore = score > 1 ? score / 100 : score;

  // Use the backend's own classification instead of recomputing it here.
  // Falls back to a score-based guess only if risk_label is ever missing
  // (e.g. an older API response), using the SAME cutoffs as the backend
  // (route_routes.py: 0.67 High / 0.34 Moderate) so the two never diverge.
  const backendLabel = route.risk_label;
  const effectiveLabel =
    backendLabel && RISK_STYLES[backendLabel]
      ? backendLabel
      : normalizedScore >= 0.67
      ? 'High'
      : normalizedScore >= 0.34
      ? 'Moderate'
      : 'Low';

  const { color, badgeClass, badgeLabel } = RISK_STYLES[effectiveLabel];

  // "Recommended" badge takes over the label text only for the safest
  // route, and only when that route isn't actually High/Moderate risk —
  // a route should never be tagged "SAFEST" if it's flagged High risk
  // (e.g. because a known high-risk hotspot sits on it).
  const displayBadgeLabel =
    route.recommended && effectiveLabel === 'Low' ? 'SAFEST' : badgeLabel;

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
        <span className={`badge ${badgeClass}`}>{displayBadgeLabel}</span>
      </div>
      <div className="route-meta">
        <span>📍 {route.distance_km} km</span>
        <span>⏱ {route.duration_min} min</span>
        <span>⚠ {(normalizedScore * 100).toFixed(0)}%</span>
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

      {/* Only show the "recommended" reassurance line if the route is
          genuinely low risk — otherwise it contradicts the badge above. */}
      {route.recommended && effectiveLabel === 'Low' && (
        <div className="recommended-tag">✓ Recommended — lowest accident risk</div>
      )}
      {route.recommended && effectiveLabel !== 'Low' && (
        <div className="recommended-tag" style={{ color: '#f59e0b' }}>
          ✓ Safest of the available options — still carries some risk
        </div>
      )}
    </div>
  );
}
