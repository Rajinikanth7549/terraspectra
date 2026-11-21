import { useEffect, useRef, useState } from 'react';
import * as maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { MapboxOverlay } from '@deck.gl/mapbox';
import { GeoJsonLayer } from '@deck.gl/layers';
import Plotly from 'plotly.js-dist-min';
import { Eye, EyeOff, Layers, Radio, ShieldCheck } from 'lucide-react';

const LEGEND_ITEMS = [
  { label: 'Healthy Crop (RGB & SWIR Normal)', color: 'rgb(34, 197, 94)' },
  { label: 'Early Fungal Blight (Pre-Symptomatic)', color: 'rgb(239, 68, 68)' },
  { label: 'Water Stress (SWIR Rise)', color: 'rgb(59, 130, 246)' },
  { label: 'Nutrient Deficiency (Chlorophyll Dip)', color: 'rgb(245, 158, 11)' },
];

export default function App() {
  const [week, setWeek] = useState(3);
  const [showAiLayer, setShowAiLayer] = useState(true);
  const [prediction, setPrediction] = useState(null);
  const [spectrum, setSpectrum] = useState(null);
  const [hovered, setHovered] = useState(null);
  const [selectedZone, setSelectedZone] = useState({
    label: 'Early Fungal Blight',
    confidence: 0.99,
    action: 'Targeted fungicide spot-spray (Red Edge/SWIR anomaly)',
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const overlayRef = useRef(null);
  const chartRef = useRef(null);
  const weekRef = useRef(week);
  const pixelPickedRef = useRef(false);

  useEffect(() => {
    weekRef.current = week;
  }, [week]);

  // 1. Initialize MapLibre + Deck.gl overlay once
  useEffect(() => {
    if (mapRef.current) return;

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: {
        version: 8,
        sources: {
          'esri-satellite': {
            type: 'raster',
            tiles: [
              'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            ],
            tileSize: 256,
            maxzoom: 16,
            attribution: '&copy; Esri, Maxar, Earthstar Geographics',
          },
          'esri-roads': {
            type: 'raster',
            tiles: [
              'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}',
            ],
            tileSize: 256,
            maxzoom: 13,
          },
          'esri-boundaries': {
            type: 'raster',
            tiles: [
              'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
            ],
            tileSize: 256,
            maxzoom: 13,
          },
        },
        layers: [
          { id: 'esri-satellite-layer', type: 'raster', source: 'esri-satellite' },
          { id: 'esri-roads-layer', type: 'raster', source: 'esri-roads' },
          { id: 'esri-boundaries-layer', type: 'raster', source: 'esri-boundaries' },
        ],
      },
      center: [78.3595, 17.461],
      zoom: 15.0,
      maxZoom: 18,
      pitch: 15,
    });
    map.addControl(new maplibregl.NavigationControl(), 'top-left');

    const overlay = new MapboxOverlay({ interleaved: false, layers: [] });
    map.addControl(overlay);

    // Clicking healthy canopy outside stressed polygons inspects that GPS coordinate
    map.on('click', async (e) => {
      if (pixelPickedRef.current) {
        pixelPickedRef.current = false;
        return;
      }
      const { lng, lat } = e.lngLat;
      try {
        const res = await fetch(`/api/spectrum/${weekRef.current}?lat=${lat}&lon=${lng}`);
        if (res.ok) {
          const data = await res.json();
          setSpectrum(data);
          setSelectedZone({
            label: 'Healthy Crop Canopy',
            confidence: 0.99,
            action: 'No intervention required — chlorophyll & SWIR curves normal',
          });
        }
      } catch (err) {
        console.error('Spectrum fetch error:', err);
      }
    });

    mapRef.current = map;
    overlayRef.current = overlay;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // 2. Fetch weekly 3D-CNN + ViT predictions & default blight hotspot spectrum
  useEffect(() => {
    let active = true;
    setLoading(true);
    setError('');

    async function loadWeekData() {
      try {
        const [predRes, specRes] = await Promise.all([
          fetch(`/api/predict/${week}`),
          fetch(`/api/spectrum/${week}`),
        ]);

        if (!predRes.ok || !specRes.ok) {
          throw new Error('Backend API unreachable. Start FastAPI on port 8000.');
        }

        const predData = await predRes.json();
        const specData = await specRes.json();

        if (!active) return;
        setPrediction(predData);
        setSpectrum(specData);
        setSelectedZone({
          label: 'Early Fungal Blight',
          confidence: 0.99,
          action: predData.summary['1'].action,
        });
      } catch (err) {
        if (active) setError(err.message);
      } finally {
        if (active) setLoading(false);
      }
    }

    loadWeekData();
    return () => {
      active = false;
    };
  }, [week]);

  // 3. Render Deck.gl GeoJSON stress polygons & 1,000-acre farm boundary
  useEffect(() => {
    if (!overlayRef.current || !prediction) return;

    const { west, south, east, north } = prediction.bounds;
    const farmBoundaryGeoJson = {
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          geometry: {
            type: 'Polygon',
            coordinates: [[
              [west, north],
              [east, north],
              [east, south],
              [west, south],
              [west, north],
            ]],
          },
          properties: { name: '1,000-Acre Hyperspectral Block' },
        },
      ],
    };

    const layers = [
      new GeoJsonLayer({
        id: 'farm-boundary',
        data: farmBoundaryGeoJson,
        filled: true,
        stroked: true,
        getFillColor: [34, 197, 94, 42],
        getLineColor: [52, 211, 153, 210],
        getLineWidth: 3,
        lineWidthMinPixels: 2,
      }),
    ];

    if (showAiLayer) {
      layers.push(
        new GeoJsonLayer({
          id: `stress-zones-week-${week}`,
          data: prediction.geojson,
          pickable: true,
          filled: true,
          stroked: false,
          getFillColor: (d) => [...d.properties.color, 205],
          onHover: (info) => {
            setHovered(info.object ? info.object.properties : null);
          },
          onClick: async (info) => {
            if (!info.object) return;
            pixelPickedRef.current = true;
            const props = info.object.properties;
            setSelectedZone({
              label: props.label,
              confidence: props.confidence,
              action: props.action,
            });
            try {
              const res = await fetch(
                `/api/spectrum/${weekRef.current}?row=${props.row}&col=${props.col}`
              );
              if (res.ok) {
                const specData = await res.json();
                setSpectrum(specData);
              }
            } catch (err) {
              console.error('Pixel spectrum error:', err);
            }
          },
          updateTriggers: {
            getFillColor: [week, showAiLayer],
          },
        })
      );
    }

    overlayRef.current.setProps({ layers });
  }, [prediction, showAiLayer, week]);

  // 4. Render 200-band Plotly hyperspectral reflectance curve
  useEffect(() => {
    if (!chartRef.current || !spectrum) return;

    const traces = [
      {
        x: spectrum.wavelengths,
        y: spectrum.healthy_reflectance,
        type: 'scatter',
        mode: 'lines',
        name: 'Healthy Crop Baseline',
        line: { color: '#22c55e', width: 2, dash: 'dash' },
      },
      {
        x: spectrum.wavelengths,
        y: spectrum.reflectance,
        type: 'scatter',
        mode: 'lines',
        name: `Selected Pixel (${spectrum.lat}°, ${spectrum.lon}°)`,
        line: { color: '#ef4444', width: 2.5 },
      },
    ];

    const layout = {
      margin: { t: 24, r: 16, b: 44, l: 50 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: '#0b1120',
      font: { color: '#cbd5e1', size: 11 },
      xaxis: {
        title: 'Wavelength (nm) — Visible (400–700) | NIR (700–1300) | SWIR (1300–2500)',
        gridcolor: '#1e293b',
      },
      yaxis: {
        title: 'Surface Reflectance',
        range: [0, 0.65],
        gridcolor: '#1e293b',
      },
      legend: {
        orientation: 'h',
        y: 1.14,
        x: 0,
      },
      shapes: [
        {
          type: 'rect',
          xref: 'x',
          yref: 'paper',
          x0: 680,
          x1: 750,
          y0: 0,
          y1: 1,
          fillcolor: 'rgba(239, 68, 68, 0.14)',
          line: { width: 0 },
        },
        {
          type: 'rect',
          xref: 'x',
          yref: 'paper',
          x0: 1350,
          x1: 1500,
          y0: 0,
          y1: 1,
          fillcolor: 'rgba(56, 189, 248, 0.12)',
          line: { width: 0 },
        },
      ],
      annotations: [
        {
          x: 715,
          y: 0.61,
          text: 'Red Edge',
          showarrow: false,
          font: { color: '#fca5a5', size: 10 },
        },
        {
          x: 1425,
          y: 0.61,
          text: 'SWIR Water Band',
          showarrow: false,
          font: { color: '#7dd3fc', size: 10 },
        },
      ],
    };

    Plotly.react(chartRef.current, traces, layout, {
      responsive: true,
      displayModeBar: false,
    });
  }, [spectrum]);

  const blightSummary = prediction?.summary?.['1'];
  const waterSummary = prediction?.summary?.['2'];

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <Radio size={22} color="#34d399" />
          <div>
            <h1 style={{ fontSize: '18px', fontWeight: 700 }}>TerraSpectra</h1>
            <span style={{ fontSize: '12px', color: '#94a3b8' }}>
              Hyperspectral Crop Disease Forecasting • 3D-CNN + ViT
            </span>
          </div>
          <span className="brand-badge">200 Bands → 30 PCA</span>
        </div>

        <div className="controls-bar">
          {[1, 2, 3, 4].map((w) => (
            <button
              key={w}
              className={`week-btn ${week === w ? 'active' : ''}`}
              onClick={() => setWeek(w)}
            >
              Week {w} {w === 3 ? '(Unseen Test)' : ''}
            </button>
          ))}

          <button
            className={`toggle-btn ${showAiLayer ? 'ai-on' : ''}`}
            onClick={() => setShowAiLayer((prev) => !prev)}
          >
            {showAiLayer ? <Eye size={16} /> : <EyeOff size={16} />}
            {showAiLayer ? 'AI Hyperspectral Layer: ON' : 'Standard RGB View (Hidden Stress)'}
          </button>
        </div>
      </header>

      {error && (
        <div
          style={{
            margin: '16px 24px 0 24px',
            padding: '12px 16px',
            background: 'rgba(239, 68, 68, 0.15)',
            border: '1px solid #ef4444',
            borderRadius: '10px',
            color: '#fecaca',
            fontSize: '13px',
          }}
        >
          {error}
        </div>
      )}

      <section className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">Monitored Field Block</div>
          <div className="kpi-value">{prediction ? `${prediction.total_acres} ac` : '—'}</div>
          <div className="kpi-sub">128 × 128 GeoTIFF (EPSG:4326)</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Early Fungal Blight</div>
          <div className="kpi-value" style={{ color: '#f87171' }}>
            {blightSummary ? `${blightSummary.acres} ac` : '—'}
          </div>
          <div className="kpi-sub">
            {blightSummary ? `${blightSummary.percentage}% of block • Invisible in RGB` : 'Loading...'}
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Water Stress Zone</div>
          <div className="kpi-value" style={{ color: '#60a5fa' }}>
            {waterSummary ? `${waterSummary.acres} ac` : '—'}
          </div>
          <div className="kpi-sub">1450nm / 1940nm SWIR moisture shift</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Fungicide Savings</div>
          <div className="kpi-value" style={{ color: '#34d399' }}>
            {prediction ? `${prediction.chemical_reduction_pct}%` : '—'}
          </div>
          <div className="kpi-sub">Precision spot-spray vs. blanket application</div>
        </div>
      </section>

      <main className="main-grid">
        <div className="panel">
          <div className="panel-header">
            <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Layers size={16} color="#38bdf8" />
              Geospatial Disease Map — Week {week}
            </span>
            <span style={{ fontSize: '12px', color: '#94a3b8' }}>
              {loading ? 'Running 3D-CNN + ViT inference...' : 'Click any pixel to inspect 200 bands'}
            </span>
          </div>

          <div className="map-wrapper">
            <div ref={mapContainerRef} className="map-container" />

            <div className="map-legend">
              {LEGEND_ITEMS.map((item) => (
                <div key={item.label} className="legend-item">
                  <span className="legend-dot" style={{ backgroundColor: item.color }} />
                  <span>{item.label}</span>
                </div>
              ))}
            </div>

            {hovered && (
              <div className="hover-tooltip">
                <div style={{ fontWeight: 700, marginBottom: '4px', color: '#38bdf8' }}>
                  {hovered.label}
                </div>
                <div>Model Confidence: {(hovered.confidence * 100).toFixed(1)}%</div>
                <div style={{ color: '#94a3b8', marginTop: '2px' }}>
                  Grid Pixel: ({hovered.row}, {hovered.col})
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <ShieldCheck size={16} color="#34d399" />
              200-Band Hyperspectral Signature (400nm – 2500nm)
            </span>
            <span style={{ fontSize: '12px', color: '#94a3b8' }}>
              {spectrum ? `Pixel (${spectrum.row}, ${spectrum.col})` : ''}
            </span>
          </div>

          <div ref={chartRef} className="chart-wrapper" />

          <div className="prescription-box">
            <div style={{ fontWeight: 700, marginBottom: '4px', color: '#f8fafc' }}>
              Zone Diagnosis: {selectedZone.label} ({(selectedZone.confidence * 100).toFixed(1)}% confidence)
            </div>
            <div style={{ color: '#cbd5e1' }}>
              <strong>Agronomic Action:</strong> {selectedZone.action}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}