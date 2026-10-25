
"""Task 1: fetch the USGS GeoJSON feed and turn it into a DataFrame."""

import pandas as pd
import requests

BASE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary" #USGS GeoJSON feed base URL


def fetch_feed(feed: str = "all_week") -> dict:
    """purpose of fetch_feed() is to fetch earthquake data from the USGS API and return the parsed GeoJSON response as a Python dictionary.”"""
    url = f"{BASE}/{feed}.geojson"
    """constructs the complete USGS API URL by combining the base URL, feed name, and GeoJSON extension.”"""

    response = requests.get(url, timeout=30)
    """sends a GET request to the USGS API to fetch earthquake data, with a timeout of 30 seconds"""
    response.raise_for_status()
    """checks the HTTP response status and raises an exception if the request fails"""

    #Check JSON
    """attempts to parse the response as JSON and raises a ValueError if the response is not valid JSON or if the feed name is invalid"""
    try:
        payload = response.json() #payload store converted JSON data from the response
    except ValueError as exc:
        raise ValueError(
            f"Invalid JSON received from USGS feed '{feed}'. "
            "Check that the feed name is valid."
        ) from exc

       #check is it response dict or not
    if not isinstance(payload, dict):
        raise ValueError("USGS feed response must be a JSON object.")

    return payload

#Convert into dataframe
def geojson_to_df(payload: dict) -> pd.DataFrame:
    """Convert GeoJSON features into one row per earthquake event."""
    rows = []

    for feature in payload.get("features", []):
        row = {
            "id": feature.get("id"),
            **feature.get("properties", {}),#get earthquake record from key features
        }
 #make rows of data for each earthquake event
 
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        
      
        row["lon"] = coordinates[0] if len(coordinates) > 0 else None
        row["lat"] = coordinates[1] if len(coordinates) > 1 else None
        row["depth_km"] = coordinates[2] if len(coordinates) > 2 else None
        """extract longitude, latitude, and depth from the coordinates list while preventing an index error if any coordinate is missing"""
        rows.append(row)

    return pd.DataFrame(rows)

