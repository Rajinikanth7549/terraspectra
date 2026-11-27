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
def predict_farm_week(
    week: int,
    lat: Optional[float] = Query(default=None),
    lon: Optional[float] = Query(default=None),
):
    if week < 1 or week > 4:
        raise HTTPException(status_code=400, detail="Week must be between 1 and 4.")

    cache_key = (
        week,
        round(lat, 5) if lat is not None else "default",
        round(lon, 5) if lon is not None else "default",
    )
    if cache_key not in prediction_cache:
        prediction_cache[cache_key] = engine.predict_week(
            week=week, center_lat=lat, center_lon=lon
        )
    return prediction_cache[cache_key]


@app.get("/api/spectrum/{week}")
def inspect_pixel_spectrum(
    week: int,
    lat: Optional[float] = Query(default=None),
    lon: Optional[float] = Query(default=None),
    row: Optional[int] = Query(default=None),
    col: Optional[int] = Query(default=None),
    farm_lat: Optional[float] = Query(default=None),
    farm_lon: Optional[float] = Query(default=None),
):
    if week < 1 or week > 4:
        raise HTTPException(status_code=400, detail="Week must be between 1 and 4.")

    cache_key = (
        week,
        round(farm_lat, 5) if farm_lat is not None else "default",
        round(farm_lon, 5) if farm_lon is not None else "default",
    )
    pred_info = prediction_cache.get(cache_key)

    if lat is None and lon is None and row is None and col is None:
        if pred_info and "default_hotspot" in pred_info:
            row = pred_info["default_hotspot"]["row"]
            col = pred_info["default_hotspot"]["col"]
        else:
            row, col = 58, 52

    healthy_r, healthy_c = 10, 10
    if pred_info and "healthy_ref" in pred_info:
        healthy_r = pred_info["healthy_ref"]["row"]
        healthy_c = pred_info["healthy_ref"]["col"]

    target = engine.get_pixel_spectrum(
        week=week,
        lat=lat,
        lon=lon,
        row=row,
        col=col,
        farm_lat=farm_lat,
        farm_lon=farm_lon,
    )
    healthy_ref = engine.get_pixel_spectrum(
        week=week,
        row=healthy_r,
        col=healthy_c,
        farm_lat=farm_lat,
        farm_lon=farm_lon,
    )
    target["healthy_reflectance"] = healthy_ref["reflectance"]
    return target
from fastapi.responses import JSONResponse
from app.prescription import build_spray_prescription


@app.get("/api/prescription/{week}")
def get_spray_prescription(
    week: int,
    min_confidence: float = 0.75,
    lat: float | None = None,
    lon: float | None = None,
):
    if week < 1 or week > 4:
        raise HTTPException(status_code=400, detail="Week must be between 1 and 4.")
    pred = engine.predict_week(week=week, center_lat=lat, center_lon=lon)
    geojson_payload = build_spray_prescription(
        pred, min_confidence=min_confidence, lat=lat, lon=lon
    )
    return JSONResponse(
        content=geojson_payload,
        headers={
            "Content-Disposition": f'attachment; filename="terraspectra_week{week}_prescription.geojson"'
        },
    )
