from fastapi.testclient import TestClient
import torch

from app.main import app
from app.model import TerraSpectraHybridNet

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["spectral_bands"] == 200
    assert "TerraSpectraHybridNet" in payload["model"]


def test_predict_unseen_week_valid():
    response = client.get("/api/predict/3")
    assert response.status_code == 200
    payload = response.json()
    assert payload["week"] == 3
    assert payload["total_acres"] > 990.0
    assert 0.0 <= payload["chemical_reduction_pct"] <= 100.0
    assert set(payload["summary"].keys()) == {"0", "1", "2", "3"}
    assert payload["geojson"]["type"] == "FeatureCollection"
    assert len(payload["geojson"]["features"]) > 0


def test_predict_week_out_of_range():
    response = client.get("/api/predict/9")
    assert response.status_code == 400
    assert "Week must be between 1 and 4" in response.json()["detail"]


def test_pixel_spectrum_endpoint():
    response = client.get("/api/spectrum/3?row=58&col=52")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["wavelengths"]) == 200
    assert len(payload["reflectance"]) == 200
    assert len(payload["healthy_reflectance"]) == 200
    assert payload["wavelengths"][0] == 400.0
    assert payload["wavelengths"][-1] == 2500.0


def test_hybrid_network_tensor_shapes():
    model = TerraSpectraHybridNet(num_classes=4)
    model.eval()
    dummy_patches = torch.randn(2, 1, 30, 11, 11)
    with torch.no_grad():
        logits = model(dummy_patches)
    assert logits.shape == (2, 4)
    assert not torch.isnan(logits).any()

def test_spray_prescription_endpoint():
    response = client.get("/api/prescription/3?min_confidence=0.75")
    assert response.status_code == 200
    assert "attachment;" in response.headers.get("content-disposition", "")
    payload = response.json()
    assert payload["type"] == "FeatureCollection"
    assert payload["metadata"]["actionable_zones"] == len(payload["features"])
    assert all(f["properties"]["confidence"] >= 0.75 for f in payload["features"])
