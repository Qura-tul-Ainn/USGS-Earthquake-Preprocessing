
"""Make predictions using the trained earthquake classifier."""

import json
from pathlib import Path

import joblib
import pandas as pd

from quakes import config


def predict_earthquake(values):    # input values from user 
    """Predict whether an earthquake belongs to the big_quake class."""

    root = config.ROOT
    model_path = root / "models" / "classification_logistic.joblib"   # load trained logistic regression pipe line
    report_path = root / "reports" / "classification_logistic.json"   # read threshold decision

    model = joblib.load(model_path)

    with report_path.open(encoding="utf-8") as file:
        report = json.load(file)

    threshold = report["threshold"]

    # Convert user values into one row of model input.
    X = pd.DataFrame([values])

    probability = float(model.predict_proba(X)[0, 1])    #estimated probability Earthquake in which class
    prediction = int(probability >= threshold)           #Check rhreshold

    return {
        "probability": probability,
        "threshold": threshold,
        "prediction": prediction,
    }


if __name__ == "__main__":   # taking input from user
    values = {
        "hour": int(input("Hour (0-23): ")),
        "dayofweek": int(input("Day (0-6): ")),
        "update_lag_hours": float(input("Update lag hours: ")),
        "is_reviewed": int(input("Is reviewed (0/1): ")),
        "nst_missing": int(input("NST missing (0/1): ")),
        "abs_lat": float(input("Absolute latitude: ")),
        "is_shallow": int(input("Is shallow (0/1): ")),
        "type": input("Type (e.g. earthquake): "),
        "region": input("Region (e.g. Other): "),
        "lat": float(input("Latitude (-90 to 90): ")),
        "lon": float(input("Longitude (-180 to 180): ")),
        "depth_km": float(input("Depth in km: ")),
    }

    result = predict_earthquake(values)

    print("\n--- Earthquake Prediction ---")
    print(f"Probability of big quake: {result['probability']:.2%}")
    print(f"Decision threshold: {result['threshold']:.2%}")
    print(f"Final prediction: {result['prediction']}")
    print("1 = big quake, 0 = not big quake")
