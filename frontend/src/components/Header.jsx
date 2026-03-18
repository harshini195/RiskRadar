import React from 'react';

export default function Header({ routes, hotspots }) {
  const highCount = hotspots.filter(h => h.risk_score >= 0.7).length;
  const modCount  = hotspots.filter(h => h.risk_score >= 0.4 && h.risk_score < 0.7).length;
  const lowCount  = hotspots.filter(h => h.risk_score < 0.4).length;

  return (
    <header className="app-header">
      <div className="logo">
        <div className="logo-icon">⚠</div>
        <div>
          <div className="logo-name">RiskRadar</div>
          <div className="logo-sub">Intelligent Accident Risk Detection</div>
        </div>
      </div>
      <div className="header-stats">
        <div className="hstat">
          <span className="live-dot" />
          Live Analysis
        </div>
        <div className="hstat">
          <span className="dot-indicator" style={{ background: '#ef4444' }} />
          {highCount} High Risk
        </div>
        <div className="hstat">
          <span className="dot-indicator" style={{ background: '#f59e0b' }} />
          {modCount} Moderate
        </div>
        <div className="hstat">
          <span className="dot-indicator" style={{ background: '#22c55e' }} />
          {lowCount} Safe
        </div>
        {routes.length > 0 && (
          <div className="hstat">
            <span className="dot-indicator" style={{ background: '#4f8ef7' }} />
            {routes.length} Routes Analyzed
          </div>
        )}
      </div>
    </header>
  );
}
