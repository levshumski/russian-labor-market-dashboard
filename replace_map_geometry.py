"""Build the project GeoJSON from the curated 85-subject TopoJSON export."""

from pathlib import Path
import json

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_FILE = PROJECT_DIR / "curated_85.geojson"
OUTPUT_FILE = PROJECT_DIR / "russia_adm1.geojson"
REGIONS_FILE = PROJECT_DIR / "rosstat_regions_for_map.csv"


def normalized_name(value):
    aliases = {
        "Кемеровская область — Кузбасс": "Кемеровская область",
        "Республика Северная Осетия — Алания": "Республика Северная Осетия-Алания",
    }
    return aliases.get(str(value).strip(), str(value).strip())


def signed_area(ring):
    """Плоская знаковая площадь кольца."""
    return sum(
        ring[index][0] * ring[(index + 1) % len(ring)][1]
        - ring[(index + 1) % len(ring)][0] * ring[index][1]
        for index in range(len(ring))
    ) / 2


def orient_polygon(rings):
    """Ориентирует внешний контур для D3, внутренние отверстия — наоборот."""
    oriented = []
    for index, ring in enumerate(rings):
        must_be_positive = index == 0
        if (signed_area(ring) > 0) != must_be_positive:
            ring = list(reversed(ring))
        oriented.append(ring)
    return oriented


def orient_geometry(geometry):
    result = dict(geometry)
    if geometry["type"] == "Polygon":
        result["coordinates"] = orient_polygon(geometry["coordinates"])
    elif geometry["type"] == "MultiPolygon":
        result["coordinates"] = [orient_polygon(polygon) for polygon in geometry["coordinates"]]
    else:
        raise ValueError(f"Неподдерживаемый тип геометрии: {geometry['type']}")
    return result


regions = pd.read_csv(REGIONS_FILE, encoding="utf-8-sig")
iso_by_name = dict(zip(regions["Субъект_РФ"], regions["Код_ISO_субъекта"]))
source = json.loads(SOURCE_FILE.read_text(encoding="utf-8"))

features = []
for feature in source["features"]:
    properties = feature.get("properties", {})
    full_name = properties.get("name_full") or properties.get("name")
    subject = normalized_name(full_name)
    shape_iso = "RU-CR" if subject == "Республика Крым" else iso_by_name.get(subject)
    if not shape_iso:
        raise ValueError(f"Нет ISO-сопоставления для геометрии: {subject}")
    features.append({
        "type": "Feature",
        "properties": {
            "shapeName": subject,
            "shapeISO": shape_iso,
            "shapeID": f"curated-{properties.get('id', shape_iso)}",
            "shapeGroup": "RUS",
            "shapeType": "ADM1",
            "center_lon": properties.get("clon"),
            "center_lat": properties.get("clat"),
        },
        "geometry": orient_geometry(feature["geometry"]),
    })

if len(features) != 85:
    raise ValueError(f"Ожидалось 85 контуров, получено {len(features)}")
if not any(item["properties"]["shapeISO"] == "RU-SEV" for item in features):
    raise ValueError("В итоговой геометрии отсутствует Севастополь")

OUTPUT_FILE.write_text(
    json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
    encoding="utf-8",
)
print(f"Карта обновлена: {len(features)} контуров, включая Севастополь")

