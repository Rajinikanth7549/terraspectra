import json
import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch

from app.model import TerraSpectraHybridNet
from app.pca_pipeline import transform_geotiff_to_pca

PATCH_SIZE = 11
PAD = PATCH_SIZE // 2
CLASS_NAMES = [
    "Healthy Crop",
    "Early Fungal Blight",
    "Water Stress",
    "Nutrient Deficiency",
]


def evaluate_unseen_week(week: int = 3):
    device = torch.device("cpu")
    scaler = joblib.load("models/scaler.pkl")
    pca = joblib.load("models/pca.pkl")

    pca_cube, _ = transform_geotiff_to_pca(f"data/farm_week{week}.tif", scaler, pca)
    y_true = np.load(f"data/labels_week{week}.npy").reshape(-1)

    C, H, W = pca_cube.shape
    padded = np.pad(pca_cube, ((0, 0), (PAD, PAD), (PAD, PAD)), mode="reflect")
    patches = np.empty((H * W, 1, C, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

    idx = 0
    for r in range(H):
        for c in range(W):
            patches[idx, 0] = padded[:, r : r + PATCH_SIZE, c : c + PATCH_SIZE]
            idx += 1

    model = TerraSpectraHybridNet(num_classes=4).to(device)
    state_dict = torch.load(
        "models/terraspectra_hybrid.pth", map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()

    preds = []
    batch_size = 512
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            batch = torch.from_numpy(patches[start : start + batch_size]).to(device)
            logits = model(batch)
            preds.append(torch.argmax(logits, dim=1).numpy())

    y_pred = np.concatenate(preds)

    overall_acc = float(accuracy_score(y_true, y_pred))
    variance_retained = float(np.sum(pca.explained_variance_ratio_) * 100.0)
    report_text = classification_report(
        y_true, y_pred, labels=[0, 1, 2, 3], target_names=CLASS_NAMES, digits=4
    )
    report_dict = classification_report(
        y_true, y_pred, labels=[0, 1, 2, 3], target_names=CLASS_NAMES, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    print(f"=== TerraSpectra Unseen Test Evaluation (Week {week}) ===")
    print(f"PCA Variance Retained (200 -> 30 bands): {variance_retained:.2f}%")
    print(f"Overall Pixel Accuracy: {overall_acc * 100:.2f}%\n")
    print("Classification Report:")
    print(report_text)
    print("Confusion Matrix (Rows=True, Cols=Predicted):")
    print(cm)

    os.makedirs("reports", exist_ok=True)
    output_path = "reports/evaluation_metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_week": week,
                "total_pixels_evaluated": int(H * W),
                "pca_components": int(C),
                "pca_variance_retained_pct": round(variance_retained, 2),
                "overall_accuracy_pct": round(overall_acc * 100.0, 4),
                "class_names": CLASS_NAMES,
                "confusion_matrix": cm.tolist(),
                "classification_report": report_dict,
            },
            f,
            indent=2,
        )
    print(f"\nSaved metrics to backend/{output_path}")


if __name__ == "__main__":
    evaluate_unseen_week(week=3)