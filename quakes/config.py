
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FALLBACK_PATH = ROOT / "data" / "fallback" / "all_week.geojson"
STREAM_DIR = ROOT / "data" / "stream"
PROCESSED_DIR = ROOT / "data" / "processed"

SEED = 42
TARGET = "big_quake"

# Numeric feature columns
NUMERIC: list[str] = [
    "hour",
    "dayofweek",
    "update_lag_hours",
    "is_reviewed",
    "nst_missing",
    "abs_lat",
    "is_shallow",
]

# Categorical feature columns (one-hot encoded)
NOMINAL: list[str] = [
    "type",
    "region",
]

# Columns that encode the magnitude: must be dropped
LEAKY: list[str] = [
    "title",
    "sig",
    "mmi",
    "cdi",
    "felt",
    "alert",
]
