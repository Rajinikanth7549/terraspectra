import os
import joblib
import numpy as np
import rasterio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

N_COMPONENTS = 30

def fit_pca_pipeline():
    os.makedirs("models", exist_ok=True)
    pixel_samples = []

    for week in range(1, 5):
        with rasterio.open(f"data/farm_week{week}.tif") as src:
            cube = src.read()  # Shape: (200, H, W)
            B, H, W = cube.shape
            pixels = cube.reshape(B, -1).T  # Shape: (H*W, 200)
            pixel_samples.append(pixels)

    all_pixels = np.vstack(pixel_samples)  # Shape: (65536, 200)

    scaler = StandardScaler()
    scaled_pixels = scaler.fit_transform(all_pixels)

    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    pca.fit(scaled_pixels)

    variance_retained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"PCA fitted: 200 bands -> {N_COMPONENTS} components")
    print(f"Total spectral variance retained: {variance_retained:.2f}%")

    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(pca, "models/pca.pkl")
    print("Saved models/scaler.pkl and models/pca.pkl")

def transform_geotiff_to_pca(tif_path, scaler=None, pca=None):
    if scaler is None:
        scaler = joblib.load("models/scaler.pkl")
    if pca is None:
        pca = joblib.load("models/pca.pkl")

    with rasterio.open(tif_path) as src:
        cube = src.read()  # (200, H, W)
        meta = {
            "bounds": src.bounds,
            "crs": str(src.crs),
            "transform": src.transform,
            "height": src.height,
            "width": src.width,
        }

    B, H, W = cube.shape
    pixels = cube.reshape(B, -1).T
    scaled = scaler.transform(pixels)
    reduced = pca.transform(scaled)  # (H*W, 30)
    pca_cube = reduced.T.reshape(N_COMPONENTS, H, W).astype(np.float32)
    return pca_cube, meta

if __name__ == "__main__":
    fit_pca_pipeline()