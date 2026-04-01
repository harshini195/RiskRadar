const BASE_URL = 'http://localhost:5000/api';

async function apiFetch(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: options.method || 'GET',   // 👈 FORCE method first
    headers: {'Content-Type': 'application/json',},
    ...(options.body && { body: options.body }), // ✅ safe
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `API error ${res.status}`);
  }
  return res.json();
}

export async function analyzeRoutes(origin, destination) {
  const [routeData, hotspotData] = await Promise.all([
    apiFetch('/routes/analyze', {
      method: 'POST',
      body: JSON.stringify({ origin, destination }),
    }),
    apiFetch(`/hotspots/?lat=12.97&lon=77.59&radius=20`),
  ]);
  return { ...routeData, hotspots: hotspotData.hotspots };
}

export async function predictRisk(segment) {
  return apiFetch('/risk/predict', {
    method: 'POST',
    body: JSON.stringify({ segment }),
  });
}

export async function getModelMetrics() {
  return apiFetch('/risk/metrics');
}

export async function getHotspots(lat, lon, radius = 20, minRisk = 0) {
  return apiFetch(`/hotspots/?lat=${lat}&lon=${lon}&radius=${radius}&min_risk=${minRisk}`);
}
