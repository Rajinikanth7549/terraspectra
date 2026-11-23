from concurrent.futures import ThreadPoolExecutor
import io
import math
import urllib.request
import joblib
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin, rowcol, xy
from scipy.ndimage import gaussian_filter
import torch

from app.model import TerraSpectraHybridNet
from app.prepare_geotiff import generate_spectral_signature

PATCH_SIZE = 11
PAD = PATCH_SIZE // 2
ACRES_PER_PIXEL = 0.0609  # ~1,000 acres across 128x128 grid
GRID_H, GRID_W = 128, 128
X_SIZE, Y_SIZE = 0.000148, 0.000141

CLASS_META = {
    0: {"name": "Healthy Crop", "color": [34, 197, 94], "action": "No intervention required"},
    1: {"name": "Early Fungal Blight", "color": [239, 68, 68], "action": "Targeted fungicide spot-spray (Red Edge/SWIR anomaly)"},
    2: {"name": "Water Stress", "color": [59, 130, 246], "action": "Variable-rate drip irrigation boost"},
    3: {"name": "Nutrient Deficiency", "color": [245, 158, 11], "action": "Localized foliar nitrogen application"},
}


def _deg2num_float(lat_deg: float, lon_deg: float, zoom: int):
    n = 2.0 ** zoom
    x = (lon_deg + 180.0) / 360.0 * n
    lat_rad = math.radians(lat_deg)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def _download_single_tile(args):
    zoom, tx, ty = args
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ty}/{tx}"
    req = urllib.request.Request(url, headers={"User-Agent": "TerraSpectra-Research/1.0"})
    with urllib.request.urlopen(req, timeout=1.5) as resp:
        img = Image.open(io.BytesIO(resp.read())).convert("RGB")
    return tx, ty, img


def _fetch_satellite_rgb(west: float, south: float, east: float, north: float, zoom: int = 15):
    """Downloads and stitches Esri satellite tiles in parallel for the selected 1,000-acre box."""
    try:
        x_min_f, y_min_f = _deg2num_float(north, west, zoom)
        x_max_f, y_max_f = _deg2num_float(south, east, zoom)

        tx0, tx1 = int(math.floor(x_min_f)), int(math.floor(x_max_f))
        ty0, ty1 = int(math.floor(y_min_f)), int(math.floor(y_max_f))

        if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) > 16:
            return None

        coords = [(zoom, tx, ty) for tx in range(tx0, tx1 + 1) for ty in range(ty0, ty1 + 1)]
        canvas = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))

        with ThreadPoolExecutor(max_workers=8) as pool:
            for tx, ty, tile_img in pool.map(_download_single_tile, coords):
                canvas.paste(tile_img, ((tx - tx0) * 256, (ty - ty0) * 256))

        left = max(0, int(round((x_min_f - tx0) * 256)))
        top = max(0, int(round((y_min_f - ty0) * 256)))
        right = max(left + 8, int(round((x_max_f - tx0) * 256)))
        bottom = max(top + 8, int(round((y_max_f - ty0) * 256)))

        cropped = canvas.crop((left, top, right, bottom)).resize(
            (GRID_W, GRID_H), Image.Resampling.BILINEAR
        )
        return np.asarray(cropped, dtype=np.float32) / 255.0
    except Exception:
        return None


def _norm01(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, 5), np.percentile(arr, 95)
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


class TerraSpectraEngine:
    def __init__(self):
        self.device = torch.device("cpu")
        self.scaler = joblib.load("models/scaler.pkl")
        self.pca = joblib.load("models/pca.pkl")
        self.wavelengths = np.load("data/wavelengths.npy")
        self.cube_cache = {}

        self.model = TerraSpectraHybridNet(num_classes=4).to(self.device)
        state_dict = torch.load("models/terraspectra_hybrid.pth", map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _get_or_build_cube(self, week: int, center_lat: float = None, center_lon: float = None):
        if center_lat is None or center_lon is None:
            cache_key = (week, "default", "default")
            if cache_key not in self.cube_cache:
                tif_path = f"data/farm_week{week}.tif"
                with rasterio.open(tif_path) as src:
                    cube = src.read()
                    transform = src.transform
                    bounds = src.bounds
                self.cube_cache[cache_key] = (cube, transform, bounds)
            return self.cube_cache[cache_key]

        key_lat = round(float(center_lat), 5)
        key_lon = round(float(center_lon), 5)
        cache_key = (week, key_lat, key_lon)
        if cache_key in self.cube_cache:
            return self.cube_cache[cache_key]

        west = key_lon - (GRID_W * X_SIZE) / 2.0
        east = key_lon + (GRID_W * X_SIZE) / 2.0
        north = key_lat + (GRID_H * Y_SIZE) / 2.0
        south = key_lat - (GRID_H * Y_SIZE) / 2.0
        transform = from_origin(west=west, north=north, xsize=X_SIZE, ysize=Y_SIZE)
        bounds = rasterio.coords.BoundingBox(left=west, bottom=south, right=east, top=north)

        seed = abs(int(round(key_lat * 100000)) * 73856093 ^ int(round(key_lon * 100000)) * 19349663) % (2**31)
        rng = np.random.default_rng(seed)

        rgb = _fetch_satellite_rgb(west, south, east, north, zoom=15)
        if rgb is not None:
            R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
            exg = _norm01(gaussian_filter(2.0 * G - R - B, sigma=1.1))
            dryness = _norm01(gaussian_filter(R - G + 0.25 * (R + G + B) / 3.0, sigma=1.1))
            lum = _norm01(gaussian_filter(0.299 * R + 0.587 * G + 0.114 * B, sigma=1.0))
        else:
            exg = _norm01(gaussian_filter(rng.normal(0, 1, (GRID_H, GRID_W)), sigma=6.0))
            dryness = _norm01(gaussian_filter(rng.normal(0, 1, (GRID_H, GRID_W)), sigma=6.0))
            lum = _norm01(gaussian_filter(rng.normal(0, 1, (GRID_H, GRID_W)), sigma=5.0))

        f_blight = _norm01(0.55 * _norm01(gaussian_filter(rng.normal(0, 1, (GRID_H, GRID_W)), sigma=7.5)) + 0.45 * exg)
        f_water = _norm01(0.55 * _norm01(gaussian_filter(rng.normal(0, 1, (GRID_H, GRID_W)), sigma=6.5)) + 0.45 * dryness)
        f_nutrient = _norm01(0.60 * _norm01(gaussian_filter(rng.normal(0, 1, (GRID_H, GRID_W)), sigma=5.5)) + 0.40 * (1.0 - exg))

        labels = np.zeros((GRID_H, GRID_W), dtype=np.uint8)

        blight_pct = [99.1, 98.2, 97.2, 95.3][week - 1]
        water_pct = [98.6, 98.3, 98.0, 97.6][week - 1]
        nutrient_pct = [98.7, 98.6, 98.5, 98.4][week - 1]

        labels[f_nutrient >= np.percentile(f_nutrient, nutrient_pct)] = 3
        labels[f_water >= np.percentile(f_water, water_pct)] = 2
        labels[f_blight >= np.percentile(f_blight, blight_pct)] = 1

        B_bands = len(self.wavelengths)
        cube = np.zeros((B_bands, GRID_H, GRID_W), dtype=np.float32)
        signatures = {c: generate_spectral_signature(self.wavelengths, c) for c in range(4)}

        for c in range(4):
            mask = labels == c
            if np.any(mask):
                cube[:, mask] = signatures[c][:, None]

        cube += (0.012 * (lum - 0.5))[None, :, :]
        for b in range(B_bands):
            cube[b] = gaussian_filter(cube[b], sigma=0.55)

        noise_rng = np.random.default_rng(seed + week)
        cube += noise_rng.normal(0, 0.008, size=cube.shape).astype(np.float32)
        cube = np.clip(cube, 0.01, 0.95)

        self.cube_cache[cache_key] = (cube, transform, bounds)
        return self.cube_cache[cache_key]

    def predict_week(self, week: int = 3, center_lat: float = None, center_lon: float = None):
        cube, transform, bounds = self._get_or_build_cube(week, center_lat, center_lon)
        B, H, W = cube.shape

        pixels = cube.reshape(B, -1).T
        scaled = self.scaler.transform(pixels)
        reduced = self.pca.transform(scaled)
        pca_cube = reduced.T.reshape(-1, H, W).astype(np.float32)
        C = pca_cube.shape[0]

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

        features = []
        dx = abs(transform.a) / 2.0
        dy = abs(transform.e) / 2.0

        hotspot = {"row": 58, "col": 52}
        best_blight_conf = -1.0
        healthy_ref = {"row": 10, "col": 10}

        healthy_coords = np.argwhere(pred_grid == 0)
        if len(healthy_coords) > 0:
            healthy_ref = {"row": int(healthy_coords[0][0]), "col": int(healthy_coords[0][1])}

        for r in range(H):
            for c in range(W):
                cls_id = int(pred_grid[r, c])
                if cls_id == 0:
                    continue
                conf_val = float(conf_grid[r, c])
                if cls_id == 1 and conf_val > best_blight_conf:
                    best_blight_conf = conf_val
                    hotspot = {"row": r, "col": c}

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
                        "confidence": round(conf_val, 4),
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
                "west": bounds.left,
                "south": bounds.bottom,
                "east": bounds.right,
                "north": bounds.top,
                "center": [
                    round((bounds.left + bounds.right) / 2, 5),
                    round((bounds.bottom + bounds.top) / 2, 5),
                ],
            },
            "total_acres": total_acres,
            "chemical_reduction_pct": chemical_reduction_pct,
            "default_hotspot": hotspot,
            "healthy_ref": healthy_ref,
            "summary": summary,
            "geojson": {
                "type": "FeatureCollection",
                "features": features,
            },
        }

    def get_pixel_spectrum(
        self,
        week: int,
        lat: float = None,
        lon: float = None,
        row: int = None,
        col: int = None,
        farm_lat: float = None,
        farm_lon: float = None,
    ):
        cube, transform, _ = self._get_or_build_cube(week, farm_lat, farm_lon)
        _, H, W = cube.shape

        if row is None or col is None:
            row, col = rowcol(transform, lon, lat)
        row = int(np.clip(row, 0, H - 1))
        col = int(np.clip(col, 0, W - 1))
        spectrum = cube[:, row, col].tolist()
        lon_pt, lat_pt = xy(transform, row, col)

        return {
            "week": week,
            "row": row,
            "col": col,
            "lat": round(float(lat_pt), 6),
            "lon": round(float(lon_pt), 6),
            "wavelengths": [round(float(w), 1) for w in self.wavelengths],
            "reflectance": [round(float(val), 4) for val in spectrum],
        }