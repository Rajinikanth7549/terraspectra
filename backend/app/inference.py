import joblib
import numpy as np
import rasterio
from rasterio.transform import rowcol, xy
import torch

from app.model import TerraSpectraHybridNet
from app.pca_pipeline import transform_geotiff_to_pca

PATCH_SIZE = 11
PAD = PATCH_SIZE // 2
ACRES_PER_PIXEL = 0.0609  # ~1,000 acres across 128x128 grid

CLASS_META = {
    0: {"name": "Healthy Crop", "color": [34, 197, 94], "action": "No intervention required"},
    1: {"name": "Early Fungal Blight", "color": [239, 68, 68], "action": "Targeted fungicide spot-spray (Red Edge/SWIR anomaly)"},
    2: {"name": "Water Stress", "color": [59, 130, 246], "action": "Variable-rate drip irrigation boost"},
    3: {"name": "Nutrient Deficiency", "color": [245, 158, 11], "action": "Localized foliar nitrogen application"},
}

class TerraSpectraEngine:
    def __init__(self):
        self.device = torch.device("cpu")
        self.scaler = joblib.load("models/scaler.pkl")
        self.pca = joblib.load("models/pca.pkl")
        self.wavelengths = np.load("data/wavelengths.npy").tolist()

        self.model = TerraSpectraHybridNet(num_classes=4).to(self.device)
        state_dict = torch.load("models/terraspectra_hybrid.pth", map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_week(self, week: int = 3):
        tif_path = f"data/farm_week{week}.tif"
        pca_cube, meta = transform_geotiff_to_pca(tif_path, self.scaler, self.pca)
        C, H, W = pca_cube.shape
        transform = meta["transform"]

        padded = np.pad(pca_cube, ((0, 0), (PAD, PAD), (PAD, PAD)), mode="reflect")
        patches = np.empty((H * W, 1, C, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

        idx = 0
        for r in range(H):
            for c in range(W):
                patches[idx, 0] = padded[:, r : r + PATCH_SIZE, c : c + PATCH_SIZE]
                idx += 1

        preds = []
        confidences = []
        batch_size = 512
        with torch.no_grad():
            for start in range(0, len(patches), batch_size):
                batch = torch.from_numpy(patches[start : start + batch_size]).to(self.device)
                logits = self.model(batch)
                probs = torch.softmax(logits, dim=1)
                conf, cls = torch.max(probs, dim=1)
                preds.append(cls.numpy())
                confidences.append(conf.numpy())

        pred_grid = np.concatenate(preds).reshape(H, W)
        conf_grid = np.concatenate(confidences).reshape(H, W)

        # Build GeoJSON polygons for stressed zones (classes 1, 2, 3) + summary stats
        features = []
        dx = abs(transform.a) / 2.0
        dy = abs(transform.e) / 2.0

        for r in range(H):
            for c in range(W):
                cls_id = int(pred_grid[r, c])
                if cls_id == 0:
                    continue
                lon, lat = xy(transform, r, c)
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [lon - dx, lat + dy],
                            [lon + dx, lat + dy],
                            [lon + dx, lat - dy],
                            [lon - dx, lat - dy],
                            [lon - dx, lat + dy],
                        ]],
                    },
                    "properties": {
                        "row": r,
                        "col": c,
                        "class_id": cls_id,
                        "label": CLASS_META[cls_id]["name"],
                        "confidence": round(float(conf_grid[r, c]), 4),
                        "color": CLASS_META[cls_id]["color"],
                        "action": CLASS_META[cls_id]["action"],
                    },
                })

        total_acres = round(H * W * ACRES_PER_PIXEL, 2)
        summary = {}
        for cls_id, info in CLASS_META.items():
            pixel_count = int(np.sum(pred_grid == cls_id))
            acres = round(pixel_count * ACRES_PER_PIXEL, 2)
            summary[str(cls_id)] = {
                "name": info["name"],
                "pixels": pixel_count,
                "acres": acres,
                "percentage": round((pixel_count / (H * W)) * 100, 2),
                "color": info["color"],
                "action": info["action"],
            }

        blight_acres = summary["1"]["acres"]
        chemical_reduction_pct = round((1.0 - (blight_acres / total_acres)) * 100, 1)

        return {
            "week": week,
            "bounds": {
                "west": meta["bounds"].left,
                "south": meta["bounds"].bottom,
                "east": meta["bounds"].right,
                "north": meta["bounds"].top,
                "center": [
                    round((meta["bounds"].left + meta["bounds"].right) / 2, 5),
                    round((meta["bounds"].bottom + meta["bounds"].top) / 2, 5),
                ],
            },
            "total_acres": total_acres,
            "chemical_reduction_pct": chemical_reduction_pct,
            "summary": summary,
            "geojson": {
                "type": "FeatureCollection",
                "features": features,
            },
        }

    def get_pixel_spectrum(self, week: int, lat: float = None, lon: float = None, row: int = None, col: int = None):
        tif_path = f"data/farm_week{week}.tif"
        with rasterio.open(tif_path) as src:
            if row is None or col is None:
                row, col = rowcol(src.transform, lon, lat)
            row = int(np.clip(row, 0, src.height - 1))
            col = int(np.clip(col, 0, src.width - 1))
            spectrum = src.read()[:, row, col].tolist()
            lon_pt, lat_pt = xy(src.transform, row, col)

        return {
            "week": week,
            "row": row,
            "col": col,
            "lat": round(float(lat_pt), 6),
            "lon": round(float(lon_pt), 6),
            "wavelengths": [round(w, 1) for w in self.wavelengths],
            "reflectance": [round(val, 4) for val in spectrum],
        }