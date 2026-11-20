from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from app.inference import TerraSpectraEngine

app = FastAPI(
    title="TerraSpectra API",
    description="Hyperspectral Crop Disease Forecasting Engine (3D-CNN + ViT)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = TerraSpectraEngine()
prediction_cache = {}

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "model": "TerraSpectraHybridNet (3D-CNN + ViT)",
        "spectral_bands": len(engine.wavelengths),
    }

@app.get("/api/predict/{week}")
def predict_farm_week(week: int):
    if week < 1 or week > 4:
        raise HTTPException(status_code=400, detail="Week must be between 1 and 4.")
    if week not in prediction_cache:
        prediction_cache[week] = engine.predict_week(week)
    return prediction_cache[week]

@app.get("/api/spectrum/{week}")
def inspect_pixel_spectrum(
    week: int,
    lat: Optional[float] = Query(default=None),
    lon: Optional[float] = Query(default=None),
    row: Optional[int] = Query(default=None),
    col: Optional[int] = Query(default=None),
):
    if week < 1 or week > 4:
        raise HTTPException(status_code=400, detail="Week must be between 1 and 4.")

    # Default to the primary fungal blight hotspot if no coordinates are passed
    if lat is None and lon is None and row is None and col is None:
        row, col = 58, 52

    target = engine.get_pixel_spectrum(week=week, lat=lat, lon=lon, row=row, col=col)
    healthy_ref = engine.get_pixel_spectrum(week=week, row=10, col=10)
    target["healthy_reflectance"] = healthy_ref["reflectance"]
    return target