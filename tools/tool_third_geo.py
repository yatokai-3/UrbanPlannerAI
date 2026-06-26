"""
GEO TOOL: real corridor length from origin/destination names.

The designer's STEP-1 LLM is unreliable at guessing distances between specific
local roads (it once said 15 km for a ~4 km corridor), and length drives cost
almost linearly — so we resolve it deterministically instead:

  geocode(origin) + geocode(destination)  ->  haversine straight-line
  road length ≈ straight-line × DETOUR_FACTOR

Uses OpenStreetMap's Nominatim (free, no API key). Falls back to None on any
failure so the caller can use its LLM/default estimate instead.
"""

import math
import time
import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "UrbanPlannerAI/1.0 (urban transport planning project)"}

# Straight-line -> road distance multiplier (urban roads wander ~30%).
DETOUR_FACTOR = 1.3

# Simple in-memory cache so repeat lookups don't re-hit the API.
_geocode_cache = {}


def _geocode(query: str):
    """Return (lat, lon) for a place name, or None. Cached + polite (1 req/s)."""
    if query in _geocode_cache:
        return _geocode_cache[query]
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": query, "format": "json", "limit": 1},
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        result = (float(data[0]["lat"]), float(data[0]["lon"])) if data else None
    except Exception as e:
        print(f"  [geo] geocode failed for '{query}': {e}")
        result = None
    _geocode_cache[query] = result
    time.sleep(1)  # Nominatim usage policy: max ~1 request/second
    return result


def _haversine_km(a, b) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    (lat1, lon1), (lat2, lon2) = a, b
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def get_corridor_length(origin: str, destination: str, city: str = "") -> dict | None:
    """
    Estimate real corridor length from endpoint names.

    Returns {"length_km", "straight_line_km", "method"} or None if either
    endpoint can't be geocoded (caller should then fall back to an estimate).
    """
    if not origin or not destination:
        return None

    suffix = f", {city}, India" if city else ", India"
    a = _geocode(f"{origin}{suffix}")
    b = _geocode(f"{destination}{suffix}")
    if not a or not b:
        return None

    straight = _haversine_km(a, b)
    return {
        "length_km": round(straight * DETOUR_FACTOR, 1),
        "straight_line_km": round(straight, 2),
        "method": f"Nominatim geocode + haversine x{DETOUR_FACTOR} detour",
    }


# if __name__ == "__main__":
#     import json
#     print(json.dumps(get_corridor_length("Ambabari", "Sindhi Camp", "Jaipur"), indent=2))
