import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './App.css';

const GMAPS_KEY = import.meta.env.VITE_GMAPS_API_KEY || '';

function loadGoogleMaps() {
  return new Promise((resolve, reject) => {
    if (window.google?.maps) { resolve(); return; }
    const script = document.createElement('script');
    script.src = `https://maps.googleapis.com/maps/api/js?key=${GMAPS_KEY}&libraries=geometry&loading=async`;
    script.async = true;
    script.onload  = resolve;
    script.onerror = reject;
    document.head.appendChild(script);
  });
}

loadGoogleMaps()
  .then(() => {
    ReactDOM.createRoot(document.getElementById('root')).render(
      <React.StrictMode><App /></React.StrictMode>
    );
  })
  .catch(() => {
    // Render app anyway — MapView degrades gracefully without Google Maps
    ReactDOM.createRoot(document.getElementById('root')).render(
      <React.StrictMode><App /></React.StrictMode>
    );
  });
