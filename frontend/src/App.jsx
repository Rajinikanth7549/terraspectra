import { useEffect, useRef, useState } from 'react';
import * as maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { MapboxOverlay } from '@deck.gl/mapbox';
import { GeoJsonLayer, ScatterplotLayer } from '@deck.gl/layers';
import Plotly from 'plotly.js-dist-min';
import { Crosshair, Download, Eye, EyeOff, Layers, MapPin, Radio, ShieldCheck } from 'lucide-react';

const LEGEND_ITEMS = [
  { label: 'Healthy Crop (RGB & SWIR Normal)', color: 'rgb(34, 197, 94)' },
  { label: 'Early Fungal Blight (Pre-Symptomatic)', color: 'rgb(239, 68, 68)' },
  { label: 'Water Stress (SWIR Rise)', color: 'rgb(59, 130, 246)' },
  { label: 'Nutrient Deficiency (Chlorophyll Dip)', color: 'rgb(245, 158, 11)' },
];

const HALF_W = (128 * 0.000148) / 2;
const HALF_H = (128 * 0.000141) / 2;

export default function App() {
  const [week, setWeek] = useState(3);
  const [showAiLayer, setShowAiLayer] = useState(true);
  const [moveFarmMode, setMoveFarmMode] = useState(false);
  const [farmCenter, setFarmCenter] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [spectrum, setSpectrum] = useState(null);
  const [hovered, setHovered] = useState(null);
  const [selectedZone, setSelectedZone] = useState({
    label: 'Early Fungal Blight',
    confidence: 0.99,
    action: 'Targeted fungicide spot-spray (Red Edge/SWIR anomaly)',
  });
  const [loading, setLoading] = useState(true);
  const [downloadingRx, setDownloadingRx] = useState(false);
  const [error, setError] = useState('');

  const handleDownloadPrescription = async () => {
    setDownloadingRx(true);
    try {
      const coords = farmCenter ? `&lat=${farmCenter.lat}&lon=${farmCenter.lon}` : '';
      const res = await fetch(`/api/prescription/${week}?min_confidence=0.75${coords}`);
      if (!res.ok) throw new Error('Failed to generate sprayer GeoJSON prescription.');
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `terraspectra_week${week}_prescription.geojson`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(err.message);
    } finally {
      setDownloadingRx(false);
    }
  };

  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const overlayRef = useRef(null);
  const chartRef = useRef(null);

  const weekRef = useRef(week);
  const predictionRef = useRef(prediction);
  const farmCenterRef = useRef(farmCenter);
  const moveFarmModeRef = useRef(moveFarmMode);
  const pixelPickedRef = useRef(false);

  useEffect(() => { weekRef.current = week; }, [week]);
  useEffect(() => { predictionRef.current = prediction; }, [prediction]);
  useEffect(() => { farmCenterRef.current = farmCenter; }, [farmCenter]);
  useEffect(() => {
    moveFarmModeRef.current = moveFarmMode;
    const map = mapRef.current;
    if (!map) return;
    map.getCanvas().style.cursor = moveFarmMode ? 'crosshair' : '';
    if (moveFarmMode) {
      map.doubleClickZoom.disable();
    } else {
      map.doubleClickZoom.enable();
    }
  }, [moveFarmMode]);

  const relocateFarmTo = (lng, lat) => {
    if (!moveFarmModeRef.current) return false;
    moveFarmModeRef.current = false;
    pixelPickedRef.current = false;
    setMoveFarmMode(false);

    // Immediately shift the green 1,000-acre boundary box to the clicked field
    const optimisticBounds = {
      west: lng - HALF_W,
      east: lng + HALF_W,
      south: lat - HALF_H,
      north: lat + HALF_H,
      center: [Number(lng.toFixed(5)), Number(lat.toFixed(5))],
    };
    setPrediction((prev) =>
      prev
        ? {
            ...prev,
            bounds: optimisticBounds,
            geojson: { type: 'FeatureCollection', features: [] },
          }
        : null
    );
    setSpectrum(null);
    setFarmCenter({ lat, lon: lng });

    if (mapRef.current) {
      mapRef.current.easeTo({
        center: [lng, lat],
        zoom: Math.max(mapRef.current.getZoom(), 14.2),
        duration: 500,
      });
    }
    return true;
  };

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
      zoom: 14.5,
      maxZoom: 18,
      pitch: 0,
    });

    map.addControl(new maplibregl.NavigationControl(), 'top-left');

    const overlay = new MapboxOverlay({
      interleaved: false,
      layers: [],
      onClick: (info) => {
        if (moveFarmModeRef.current && info?.coordinate) {
          relocateFarmTo(info.coordinate[0], info.coordinate[1]);
        }
      },
    });
    map.addControl(overlay);

    map.on('click', async (e) => {
      const { lng, lat } = e.lngLat;

      if (relocateFarmTo(lng, lat)) {
        return;
      }

      if (pixelPickedRef.current) {
        pixelPickedRef.current = false;
        return;
      }

      const pred = predictionRef.current;
      if (!pred) return;
      const { west, south, east, north } = pred.bounds;

      if (lng < west || lng > east || lat < south || lat > north) {
        setSelectedZone({
          label: 'Outside Monitored Farm Block',
          confidence: 1.0,
          action: 'Click inside the green square to inspect crops, or click "Move Farm Block" to analyze here',
        });
        return;
      }

      const col = Math.min(127, Math.max(0, Math.floor(((lng - west) / (east - west)) * 128)));
      const row = Math.min(127, Math.max(0, Math.floor(((north - lat) / (north - south)) * 128)));

      const matchedFeature = pred.geojson?.features?.find(
        (f) => f.properties.row === row && f.properties.col === col
      );

      if (matchedFeature) {
        setSelectedZone({
          label: matchedFeature.properties.label,
          confidence: matchedFeature.properties.confidence,
          action: matchedFeature.properties.action,
        });
      } else {
        setSelectedZone({
          label: 'Healthy Crop Canopy',
          confidence: 0.99,
          action: 'No intervention required — chlorophyll & SWIR curves normal',
        });
      }

      try {
        const fc = farmCenterRef.current;
        const farmQuery = fc ? `&farm_lat=${fc.lat}&farm_lon=${fc.lon}` : '';
        const res = await fetch(
          `/api/spectrum/${weekRef.current}?row=${row}&col=${col}${farmQuery}`
        );
        if (res.ok) {
          const data = await res.json();
          setSpectrum(data);
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

  // 2. Fetch live 3D-CNN + ViT predictions & hotspot spectrum whenever week or farmCenter changes
  useEffect(() => {
    let active = true;
    setLoading(true);
    setError('');

    async function loadWeekData() {
      try {
        const predUrl = farmCenter
          ? `/api/predict/${week}?lat=${farmCenter.lat}&lon=${farmCenter.lon}`
          : `/api/predict/${week}`;

        const predRes = await fetch(predUrl);
        if (!predRes.ok) {
          throw new Error('Backend API unreachable. Start FastAPI on port 8000.');
        }
        const predData = await predRes.json();

        const hotspot = predData.default_hotspot || { row: 58, col: 52 };
        const farmQuery = farmCenter
          ? `&farm_lat=${farmCenter.lat}&farm_lon=${farmCenter.lon}`
          : '';
        const specRes = await fetch(
          `/api/spectrum/${week}?row=${hotspot.row}&col=${hotspot.col}${farmQuery}`
        );
        if (!specRes.ok) {
          throw new Error('Failed to fetch hyperspectral pixel signature.');
        }
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
  }, [week, farmCenter]);

  // 3. Render Deck.gl GeoJSON stress polygons, farm boundary, and selected spot pin
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
        getFillColor: [34, 197, 94, 32],
        getLineColor: [52, 211, 153, 225],
        getLineWidth: 3,
        lineWidthMinPixels: 2,
      }),
    ];

    if (showAiLayer && prediction.geojson?.features?.length > 0) {
      layers.push(
        new GeoJsonLayer({
          id: `stress-zones-${week}-${west.toFixed(4)}-${north.toFixed(4)}`,
          data: prediction.geojson,
          pickable: true,
          filled: true,
          stroked: false,
          getFillColor: (d) => [...d.properties.color, 205],
          onHover: (info) => {
            setHovered(info.object ? info.object.properties : null);
          },
          onClick: async (info) => {
            if (moveFarmModeRef.current && info?.coordinate) {
              relocateFarmTo(info.coordinate[0], info.coordinate[1]);
              return;
            }
            if (!info.object) return;
            pixelPickedRef.current = true;
            const props = info.object.properties;
            setSelectedZone({
              label: props.label,
              confidence: props.confidence,
              action: props.action,
            });
            try {
              const fc = farmCenterRef.current;
              const farmQuery = fc ? `&farm_lat=${fc.lat}&farm_lon=${fc.lon}` : '';
              const res = await fetch(
                `/api/spectrum/${weekRef.current}?row=${props.row}&col=${props.col}${farmQuery}`
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
            getFillColor: [week, showAiLayer, west, north],
          },
        })
      );
    }

    if (spectrum) {
      layers.push(
        new ScatterplotLayer({
          id: 'selected-spot-pin',
          data: [{ position: [spectrum.lon, spectrum.lat] }],
          getPosition: (d) => d.position,
          getFillColor: [56, 189, 248, 220],
          getLineColor: [255, 255, 255, 255],
          stroked: true,
          filled: true,
          radiusMinPixels: 7,
          radiusMaxPixels: 12,
          lineWidthMinPixels: 2.5,
        })
      );
    }

    overlayRef.current.setProps({ layers });
  }, [prediction, showAiLayer, week, spectrum]);

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

          <button
            className="toggle-btn"
            onClick={handleDownloadPrescription}
            disabled={downloadingRx}
            title="Download confidence-filtered GeoJSON for variable-rate sprayers"
          >
            <Download size={16} />
            {downloadingRx ? 'Exporting...' : 'Export Spray GeoJSON'}
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

            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <button
                onClick={() => setMoveFarmMode((prev) => !prev)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '5px 10px',
                  borderRadius: '8px',
                  border: moveFarmMode ? '1px solid #38bdf8' : '1px solid #334155',
                  background: moveFarmMode ? 'rgba(56, 189, 248, 0.2)' : '#0f172a',
                  color: moveFarmMode ? '#7dd3fc' : '#cbd5e1',
                  fontSize: '12px',
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                {moveFarmMode ? <Crosshair size={14} /> : <MapPin size={14} />}
                {moveFarmMode ? 'Click Target Field on Map...' : 'Move Farm Block'}
              </button>

              <span style={{ fontSize: '12px', color: '#94a3b8' }}>
                {loading ? 'Analyzing field terrain...' : 'Click inside green block'}
              </span>
            </div>
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