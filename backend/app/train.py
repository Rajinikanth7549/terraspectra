import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.pca_pipeline import transform_geotiff_to_pca
from app.model import TerraSpectraHybridNet

PATCH_SIZE = 11
PAD = PATCH_SIZE // 2

def extract_balanced_patches(weeks=(1, 2, 4), max_per_class=600):
    np.random.seed(42)
    X_list, y_list = [], []

    for week in weeks:
        pca_cube, _ = transform_geotiff_to_pca(f"data/farm_week{week}.tif")
        labels = np.load(f"data/labels_week{week}.npy")
        padded = np.pad(pca_cube, ((0, 0), (PAD, PAD), (PAD, PAD)), mode="reflect")

        for cls in range(4):
            coords = np.argwhere(labels == cls)
            if len(coords) == 0:
                continue
            sample_count = min(len(coords), max_per_class // len(weeks))
            chosen = coords[np.random.choice(len(coords), size=sample_count, replace=False)]

            for r, c in chosen:
                patch = padded[:, r : r + PATCH_SIZE, c : c + PATCH_SIZE]
                X_list.append(patch[np.newaxis, ...])  # (1, 30, 11, 11)
                y_list.append(cls)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y

def train_model():
    print("Extracting balanced 3D hyperspectral patches...")
    X, y = extract_balanced_patches(weeks=(1, 2, 4), max_per_class=600)
    print(f"Dataset shape: {X.shape} | Labels: {y.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=64,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=64,
        shuffle=False,
    )

    device = torch.device("cpu")
    model = TerraSpectraHybridNet(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 8
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x).argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        val_acc = (correct / total) * 100
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/terraspectra_hybrid.pth")
    print("Saved trained model to models/terraspectra_hybrid.pth")

if __name__ == "__main__":
    train_model()