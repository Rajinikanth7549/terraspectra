from datetime import datetime, timezone
from typing import Optional

DOSAGE_PER_ACRE_LITERS = {
    1: ("Targeted Fungicide (Azoxystrobin)", "HIGH", 1.2),
    2: ("Variable-Rate Drip Irrigation Boost", "MEDIUM", 4500.0),
    3: ("Foliar Nitrogen (UAN 32%) Spot-Feed", "MEDIUM", 3.5),
}


def build_spray_prescription(
    prediction: dict,
    min_confidence: float = 0.75,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> dict:
    """Filters high-confidence stress polygons into a sprayer-ready GeoJSON prescription."""
    actionable_features = []
    for feat in prediction["geojson"]["features"]:
        props = feat["properties"]
        cls_id = props["class_id"]
        conf = props["confidence"]
        if cls_id == 0 or conf < min_confidence:
            continue

        treatment, priority, rate_per_acre = DOSAGE_PER_ACRE_LITERS[cls_id]
        est_volume = round(0.0609 * rate_per_acre, 3)

        actionable_features.append({
            "type": "Feature",
            "geometry": feat["geometry"],
            "properties": {
                **props,
                "priority": priority,
                "treatment_agent": treatment,
                "area_acres": 0.0609,
                "recommended_volume_liters": est_volume,
            },
        })

    return {
        "type": "FeatureCollection",
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "week": prediction["week"],
            "min_confidence_threshold": min_confidence,
            "actionable_zones": len(actionable_features),
            "chemical_reduction_pct": prediction["chemical_reduction_pct"],
            "coordinates": {"lat": lat, "lon": lon},
        },
        "features": actionable_features,
    }
