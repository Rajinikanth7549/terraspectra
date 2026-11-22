# TerraSpectra — Hyperspectral Crop Disease Forecasting

**TerraSpectra** is an end-to-end geospatial AI platform that detects pre-symptomatic crop stress and fungal blight across a 1,000-acre agricultural block using **200-band hyperspectral imagery (400 nm – 2500 nm)**, **PCA dimensionality reduction**, and a **Hybrid 3D-CNN + Vision Transformer (ViT)** architecture.

---

## Key Capabilities

* **Pre-Symptomatic Disease Detection:** Identifies cellular chlorophyll breakdown in the **Red Edge (`680–750 nm`)** and leaf moisture shifts in the **SWIR Water Bands (`1350–1500 nm` / `1940 nm`)** weeks before symptoms become visible in standard RGB satellite imagery.
* **Spatial-Spectral Deep Learning (`TerraSpectraHybridNet`):** Compresses 200 hyperspectral bands down to **30 principal components** (~70.81% variance retained) and classifies patches using a PyTorch **3D-CNN + Vision Transformer** hybrid network (~99.73% validation accuracy).
* **4-Week Outbreak Progression & Precision Savings:** Tracks fungal blight expansion from **1.28 acres (Week 1)** to **13.7 acres (Week 4)** across a `128 × 128` GeoTIFF (`EPSG:4326`) block and calculates spot-spray fungicide reduction percentages.
* **Interactive Geospatial & Spectral Dashboard:** Built with **React, Vite, MapLibre GL, Deck.gl, and Plotly**—featuring hybrid satellite + administrative boundary overlays, an RGB vs. AI layer toggle, and click-to-inspect 200-band pixel reflectance curves.

---

## System Architecture

```text
200-Band GeoTIFF Cube (400nm–2500nm, 128×128, EPSG:4326)
        │
        ▼
PCA Dimensionality Reduction (200 bands → 30 components)
        │
        ▼
TerraSpectraHybridNet (3D-CNN Spatial-Spectral Extractor + ViT Encoder)
        │
        ▼
FastAPI Inference Server (/api/predict/{week}, /api/spectrum/{week})
        │
        ▼
React + MapLibre + Deck.gl + Plotly Dashboard
```

---

## Project Structure

```text
terraspectra/
├── backend/
│   ├── app/
│   │   ├── prepare_geotiff.py   # Synthetic 4-week hyperspectral raster & label generator
│   │   ├── pca_pipeline.py      # 200-to-30 band PCA transformation pipeline
│   │   ├── model.py             # PyTorch 3D-CNN + Vision Transformer (TerraSpectraHybridNet)
│   │   ├── train.py             # Model training & checkpoint export (terraspectra_hybrid.pth)
│   │   ├── inference.py         # GeoJSON stress polygon & 200-band spectrum extraction
│   │   └── main.py              # FastAPI REST endpoints
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx              # MapLibre + Deck.gl map, KPI cards, & Plotly spectral chart
    │   └── index.css            # Dark geospatial UI styling
    ├── vite.config.js           # Dev server & /api proxy configuration
    └── package.json
```

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Verifies FastAPI server and trained model availability |
| `GET` | `/api/predict/{week}` | Returns GeoJSON disease polygons, acreage breakdown, and fungicide savings for `week` (`1–4`) |
| `GET` | `/api/spectrum/{week}` | Returns 200-band reflectance curve vs. healthy baseline by grid `(row, col)` or `(lat, lon)` |

---

## Running Locally

### 1. Start the FastAPI Backend (Terminal 1)
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2. Start the React + Vite Frontend (Terminal 2)
```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` (or forward port `5173` in GitHub Codespaces) to explore the dashboard.