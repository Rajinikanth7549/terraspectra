import json
import os
import matplotlib.pyplot as plt
import numpy as np

from app.prepare_geotiff import generate_spectral_signature

ACRES_PER_PIXEL = 0.0609
CLASS_NAMES = [
    "Healthy Crop",
    "Early Fungal Blight",
    "Water Stress",
    "Nutrient Deficiency",
]


def build_all_charts():
    os.makedirs("../docs", exist_ok=True)

    # 1. Unseen Week 3 Confusion Matrix
    with open("reports/evaluation_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    cm = np.array(metrics["confusion_matrix"])
    acc = metrics["overall_accuracy_pct"]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right", fontsize=10)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax.set_xlabel("Predicted Class (TerraSpectraHybridNet)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ground Truth Class", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Unseen Week 3 Confusion Matrix (Overall Accuracy: {acc:.2f}%)",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white" if cm[i, j] > thresh else "#0f172a",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("../docs/confusion_matrix.png", dpi=220)
    plt.close(fig)

    # 2. 200-Band Hyperspectral Signatures (400nm - 2500nm)
    wavelengths = np.load("data/wavelengths.npy")
    colors = ["#16a34a", "#dc2626", "#2563eb", "#d97706"]
    styles = ["-", "--", "-.", ":"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for cls_id, (name, color, ls) in enumerate(zip(CLASS_NAMES, colors, styles)):
        sig = generate_spectral_signature(wavelengths, cls_id)
        ax.plot(wavelengths, sig, label=name, color=color, linestyle=ls, linewidth=2.2)

    ax.axvspan(680, 750, color="#ef4444", alpha=0.12, label="Red Edge (680–750 nm)")
    ax.axvspan(1350, 1500, color="#0284c7", alpha=0.12, label="SWIR Water Band (1350–1500 nm)")
    ax.set_xlabel("Wavelength (nm)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Surface Reflectance", fontsize=11, fontweight="bold")
    ax.set_title(
        "200-Band Hyperspectral Reflectance Signatures Across Crop Health States",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylim(0, 0.62)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9.5, frameon=True)
    plt.tight_layout()
    plt.savefig("../docs/spectral_signatures.png", dpi=220)
    plt.close(fig)

    # 3. 4-Week Outbreak Expansion & Fungicide Savings
    weeks = ["Week 1", "Week 2", "Week 3\n(Unseen)", "Week 4"]
    blight_acres, water_acres, nutrient_acres = [], [], []

    for w in range(1, 5):
        labels = np.load(f"data/labels_week{w}.npy")
        blight_acres.append(round(float(np.sum(labels == 1) * ACRES_PER_PIXEL), 2))
        water_acres.append(round(float(np.sum(labels == 2) * ACRES_PER_PIXEL), 2))
        nutrient_acres.append(round(float(np.sum(labels == 3) * ACRES_PER_PIXEL), 2))

    x = np.arange(len(weeks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x - width, blight_acres, width, label="Early Fungal Blight", color="#ef4444")
    ax.bar(x, water_acres, width, label="Water Stress", color="#3b82f6")
    ax.bar(x + width, nutrient_acres, width, label="Nutrient Deficiency", color="#f59e0b")

    ax.set_xticks(x)
    ax.set_xticklabels(weeks, fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Affected Area (Acres out of 997.79 ac)", fontsize=11, fontweight="bold")
    ax.set_title(
        "4-Week Crop Stress Progression & Precision Spot-Spray Target Area",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("../docs/weekly_progression.png", dpi=220)
    plt.close(fig)

    print("Generated docs/confusion_matrix.png, docs/spectral_signatures.png, and docs/weekly_progression.png")


if __name__ == "__main__":
    build_all_charts()
