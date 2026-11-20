import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

def generate_spectral_signature(wavelengths, crop_class):
    """Generates realistic 200-band reflectance curve (400nm - 2500nm)."""
    blue_red = 0.06 + 0.05 * np.exp(-((wavelengths - 550) ** 2) / (2 * 25 ** 2))
    red_edge = 0.42 / (1.0 + np.exp(-(wavelengths - 715) / 12.0))
    nir_decay = -0.18 / (1.0 + np.exp(-(wavelengths - 1350) / 80.0))
    water_dip1 = -0.14 * np.exp(-((wavelengths - 1450) ** 2) / (2 * 35 ** 2))
    water_dip2 = -0.12 * np.exp(-((wavelengths - 1940) ** 2) / (2 * 40 ** 2))
    base = blue_red + red_edge + nir_decay + water_dip1 + water_dip2

    if crop_class == 0:  # Healthy Crop
        return base
    elif crop_class == 1:  # Early Fungal Blight (RGB looks healthy; Red Edge & SWIR shift)
        blight_shift = -0.11 / (1.0 + np.exp(-(wavelengths - 705) / 15.0))
        swir_rise = 0.09 / (1.0 + np.exp(-(wavelengths - 1400) / 90.0))
        return base + blight_shift + swir_rise
    elif crop_class == 2:  # Water Stress (Strong SWIR rise, mild NIR drop)
        nir_drop = -0.05 / (1.0 + np.exp(-(wavelengths - 750) / 30.0))
        swir_rise = 0.14 / (1.0 + np.exp(-(wavelengths - 1380) / 70.0))
        return base + nir_drop + swir_rise
    elif crop_class == 3:  # Nutrient Deficiency (Chlorophyll dip in visible + lower NIR)
        vis_rise = 0.06 * np.exp(-((wavelengths - 620) ** 2) / (2 * 50 ** 2))
        nir_drop = -0.08 / (1.0 + np.exp(-(wavelengths - 730) / 25.0))
        return base + vis_rise + nir_drop
    return base

def create_farm_dataset():
    os.makedirs("data", exist_ok=True)
    np.random.seed(42)

    H, W, B = 128, 128, 200  # ~1,000 acres at ~15.7m pixel resolution
    wavelengths = np.linspace(400, 2500, B)
    np.save("data/wavelengths.npy", wavelengths)

    # Spatial GPS metadata (1,000-acre agricultural block near Hyderabad)
    transform = from_origin(west=78.3500, north=17.4700, xsize=0.000148, ysize=0.000141)
    crs = "EPSG:4326"

    # Generate 4 weekly progression snapshots (Week 1 to Week 4)
    blight_radii = [2.5, 4.0, 5.2, 8.5]  # Week 3 = ~5.2-acre fungal blight outbreak

    for week_idx, radius in enumerate(blight_radii, start=1):
        labels = np.zeros((H, W), dtype=np.uint8)
        y_grid, x_grid = np.ogrid[:H, :W]

        # Zone 2: Water Stress patch (top-right)
        water_mask = ((x_grid - 95) ** 2 + (y_grid - 32) ** 2) <= (6.5 + week_idx * 0.5) ** 2
        labels[water_mask] = 2

        # Zone 3: Nutrient Deficiency patch (bottom-left)
        nutrient_mask = ((x_grid - 35) ** 2 + (y_grid - 92) ** 2) <= 7.0 ** 2
        labels[nutrient_mask] = 3

        # Zone 1: Early Fungal Blight outbreak (center-left, ~5 acres in Week 3)
        blight_mask = ((x_grid - 52) ** 2 + (y_grid - 58) ** 2) <= radius ** 2
        labels[blight_mask] = 1

        # Build (B, H, W) hyperspectral data cube
        cube = np.zeros((B, H, W), dtype=np.float32)
        signatures = {c: generate_spectral_signature(wavelengths, c) for c in range(4)}

        for c in range(4):
            mask = (labels == c)
            if np.any(mask):
                cube[:, mask] = signatures[c][:, None]

        # Smooth spatial boundaries & add realistic sensor noise
        for b in range(B):
            cube[b] = gaussian_filter(cube[b], sigma=0.6)
        cube += np.random.normal(0, 0.008, size=cube.shape).astype(np.float32)
        cube = np.clip(cube, 0.01, 0.95)

        tif_path = f"data/farm_week{week_idx}.tif"
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=H,
            width=W,
            count=B,
            dtype="float32",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(cube)

        np.save(f"data/labels_week{week_idx}.npy", labels)
        blight_acres = round(np.sum(labels == 1) * 0.0609, 2)
        print(f"Created {tif_path} | Fungal Blight Zone: {blight_acres} acres")

if __name__ == "__main__":
    create_farm_dataset()