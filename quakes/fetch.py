
"""Task 1: fetch the USGS GeoJSON feed and turn it into a DataFrame."""

import pandas as pd
import requests

BASE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"


def fetch_feed(feed: str = "all_week") -> dict:
    """Fetch a USGS GeoJSON feed and return the parsed dictionary."""
    url = f"{BASE}/{feed}.geojson"

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError(
            f"Invalid JSON received from USGS feed '{feed}'. "
            "Check that the feed name is valid."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("USGS feed response must be a JSON object.")

    return payload


def geojson_to_df(payload: dict) -> pd.DataFrame:
    """Convert GeoJSON features into one row per earthquake event."""
    rows = []

    for feature in payload.get("features", []):
        row = {
            "id": feature.get("id"),
            **feature.get("properties", {}),
        }

        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []

        row["lon"] = coordinates[0] if len(coordinates) > 0 else None
        row["lat"] = coordinates[1] if len(coordinates) > 1 else None
        row["depth_km"] = coordinates[2] if len(coordinates) > 2 else None

        rows.append(row)

    return pd.DataFrame(rows)
