"""Task 4: feature engineering."""
import pandas as pd
from quakes import config


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'hour' and 'dayofweek' (UTC) from 'time'."""
    out = df.copy()
    if "time" not in out.columns:
        return out

    if pd.api.types.is_numeric_dtype(out["time"]):
        time = pd.to_datetime(
            out["time"], unit="ms", utc=True, errors="coerce"
        )
    else:
        time = pd.to_datetime(
            out["time"],utc=True, errors="coerce"
        )

    out["hour"] = time.dt.hour
    out["dayofweek"] = time.dt.dayofweek

    return out 
        
#abc

def add_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add:
    update_lag_hours = hours between 'time' and 'updated'
    is_reviewed      = 1 if status == 'reviewed' else 0
    nst_missing      = 1 if 'nst' is missing else 0
    """
    out = df.copy()

    if "time" in out.columns and "updated" in out.columns:
        event_time = pd.to_datetime(
            out["time"], unit="ms", utc=True, errors="coerce"
        ) if pd.api.types.is_numeric_dtype(out["time"]) else pd.to_datetime(
            out["time"], utc=True, errors="coerce"
        )

        updated_time = pd.to_datetime(
            out["updated"], unit="ms", utc=True, errors="coerce"
        ) if pd.api.types.is_numeric_dtype(out["updated"]) else pd.to_datetime(
            out["updated"], utc=True, errors="coerce"
        )

        out["update_lag_hours"] = (
            updated_time - event_time
        ).dt.total_seconds() / 3600

    if "status" in out.columns:
        out["is_reviewed"] = (
            out["status"].astype("string").str.strip().str.lower()
            .eq("reviewed").astype(int)
        )
    else:
        out["is_reviewed"] = 0

    if "nst" in out.columns:
        out["nst_missing"] = out["nst"].isna().astype(int)
    else:
        out["nst_missing"] = 1

    return out



def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'abs_lat' and 'is_shallow' (1 if depth_km < 70 else 0)."""
    out = df.copy()

    if "lat" in out.columns:
        out["abs_lat"] = pd.to_numeric(
            out["lat"], errors="coerce"
        ).abs()

    if "depth_km" in out.columns:
        depth = pd.to_numeric(out["depth_km"], errors="coerce")
        out["is_shallow"] = depth.lt(70).fillna(False).astype(int)

    return out


def group_rare(s: pd.Series, top_k: int = 15) -> pd.Series:
    """Keep the top_k most frequent values; replace all others with 'Other'."""
    if top_k < 0:
        raise ValueError("top_k must be non-negative")

    frequent_values = s.value_counts().head(top_k).index
    result = s.astype("object").copy()

    rare_mask = result.notna() & ~result.isin(frequent_values)
    result.loc[rare_mask] = "Other"

    return result



def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'big_quake' = 1 if mag >= 4.5 else 0."""
    out = df.copy()

    if "mag" not in out.columns:
        raise KeyError("Required column 'mag' is missing")

    magnitude = pd.to_numeric(out["mag"], errors="coerce")
    out["big_quake"] = magnitude.ge(4.5).fillna(False).astype(int)

    return out


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop every column listed in config.LEAKY (ignore ones that are absent)."""
    out = df.copy()
    leaky_columns = getattr(config, "LEAKY", [])
    return out.drop(columns=list(leaky_columns), errors="ignore")

