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


def get_nearby_named_places(lat, lon, radius=5000):
    query = f"""
    [out:json];
    (
      node["amenity"~"school|hospital|college|university|clinic|doctors"](around:{radius},{lat},{lon});
      way["amenity"~"school|hospital|college|university|clinic|doctors"](around:{radius},{lat},{lon});

      node["shop"~"supermarket|mall|convenience|grocery|department_store"](around:{radius},{lat},{lon});
      way["shop"~"supermarket|mall|convenience|grocery|department_store"](around:{radius},{lat},{lon});

      node["amenity"="fuel"](around:{radius},{lat},{lon});
      way["amenity"="fuel"](around:{radius},{lat},{lon});

      node["amenity"~"market|marketplace"](around:{radius},{lat},{lon});
      way["amenity"~"market|marketplace"](around:{radius},{lat},{lon});

      node["highway"="bus_stop"](around:{radius},{lat},{lon});
      node["railway"~"station|subway_entrance"](around:{radius},{lat},{lon});

      node["place"~"city|town|village"](around:{radius},{lat},{lon});
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
        # name = tags.get("name")
        name = tags.get("name", "Nearby Facility")
        amenity = tags.get("amenity")

        if not name:
            continue
         # 🔹 Determine type
        # place_type = (
        #     tags.get("amenity")
        #     or tags.get("shop")
        #     or tags.get("railway")
        #     or tags.get("highway")
        #     or "other"
        # )
        if tags.get("amenity") in ["school"]:
            place_type = "school"
        elif tags.get("amenity") in ["hospital", "clinic", "doctors"]:
            place_type = "hospital"
        elif tags.get("amenity") in ["college", "university"]:
            place_type = "college"
        elif tags.get("amenity") == "fuel":
            place_type = "petrol_bunk"
        elif tags.get("amenity") in ["market", "marketplace"]:
            place_type = "market"
        elif tags.get("shop") in ["supermarket", "mall", "convenience", "grocery", "department_store"]:
            place_type = "market"
        elif tags.get("place") in ["city", "town", "village"]:
            place_type = "access_city"
        elif tags.get("highway") == "bus_stop":
            place_type = "transit"
        elif tags.get("railway") in ["station", "subway_entrance"]:
            place_type = "transit"
        else:
            place_type = "other"


        if el["type"] == "node":
            plat, plon = el["lat"], el["lon"]
        else:
            center = el.get("center")
            if not center:
                continue
            plat, plon = center["lat"], center["lon"]

        distance_km = round(haversine(lat, lon, plat, plon), 2)

        # places.append({
        #     "name": name,
        #     "type": amenity,
        #     "distance_km": distance_km
        # })
        places.append({
            "name": name,
            "type": place_type,
            "distance_km": distance_km
        })


    places.sort(key=lambda x: x["distance_km"])
    return places
