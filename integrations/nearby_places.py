import requests
import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def get_nearby_named_places(lat, lon, radius=3000):
    query = f"""
    [out:json];
    (
      node["amenity"~"school|hospital|college"](around:{radius},{lat},{lon});
      way["amenity"~"school|hospital|college"](around:{radius},{lat},{lon});
    );
    out center tags;
    """

    url = "https://overpass-api.de/api/interpreter"

    try:
        response = requests.post(
            url,
            data=query,
            headers={"User-Agent": "GeoAI-Student-Project"},
            timeout=30
        )

        if response.status_code != 200 or not response.text.strip().startswith("{"):
            return []

        data = response.json()

    except Exception:
        return []

    places = []

    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name")
        amenity = tags.get("amenity")

        if not name:
            continue

        if el["type"] == "node":
            plat, plon = el["lat"], el["lon"]
        else:
            center = el.get("center")
            if not center:
                continue
            plat, plon = center["lat"], center["lon"]

        distance_km = round(haversine(lat, lon, plat, plon), 2)

        places.append({
            "name": name,
            "type": amenity,
            "distance_km": distance_km
        })

    places.sort(key=lambda x: x["distance_km"])
    return places
