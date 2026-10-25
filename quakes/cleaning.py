"""Task 3: cleaning earthquake data."""

import pandas as pd


def epoch_ms_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert time and updated from epoch milliseconds to UTC datetimes."""
    out = df.copy()

    for col in ("time", "updated"):
        if col in out.columns:
            if not pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = pd.to_datetime(
                    out[col], unit="ms", utc=True, errors="coerce"
                )

    return out


def dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest record for each earthquake ID."""
    out = df.copy()

    if "id" not in out.columns or "updated" not in out.columns:
        return out

    out = out.sort_values("updated", ascending=False, kind="stable")
    out = out.drop_duplicates(subset="id", keep="first")

    return out.reset_index(drop=True)


def keep_earthquakes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows whose type is earthquake, ignoring case and whitespace."""
    if "type" not in df.columns:
        return df.copy()

    mask = (
        df["type"]
        .astype("string")
        .str.strip()
        .str.lower()
        .eq("earthquake")
    )
    """keep_earthquakes() function filters the DataFrame to keep only earthquake records, ignoring uppercase letters, lowercase letters, and extra spaces"""

    return df.loc[mask].copy() # returns only the rows where the mask is True


def extract_region(series: pd.Series) -> pd.Series:
    """Extract the final comma-separated region from a place description.10 km NE of Lahore, Pakistan" → "Pakistan"""
    def get_region(value):
        if pd.isna(value):
            return value

        parts = str(value).split(",")

        if len(parts) > 1:
            return parts[-1].strip()

        return str(value).strip()

    return series.apply(get_region)


def iqr_outlier_mask(series: pd.Series) -> pd.Series:
    """Return True for values outside the 1.5 × IQR bounds."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return (series < lower_bound) | (series > upper_bound)


def drop_missing_target(
    df: pd.DataFrame, target: str = "mag"
) -> pd.DataFrame:
    """Remove rows with missing target values."""
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' is missing")

    return df.dropna(subset=[target]).copy()