
"""Part 2, Task A: extract, transform, load, with logging."""

import argparse
import json
import logging
import sqlite3

import pandas as pd

from . import cleaning, config, features
from .fetch import fetch_feed, geojson_to_df

log = logging.getLogger("quakes.etl")

DB_PATH = config.PROCESSED_DIR / "quakes.db"
LOG_PATH = config.ROOT / "logs" / "etl.log"


def extract(feed: str = "all_week") -> pd.DataFrame:
    """Fetch earthquake data, falling back to the local GeoJSON file."""
    try:
        payload = fetch_feed(feed)
        df = geojson_to_df(payload)
        log.info("Extracted %d rows from the USGS feed", len(df))
        return df

    except Exception as exc:
        log.warning("USGS feed failed: %s. Trying fallback file.", exc)

        if not config.FALLBACK_PATH.exists():
            raise FileNotFoundError(
                f"Fallback file not found: {config.FALLBACK_PATH}"
            ) from exc

        with open(config.FALLBACK_PATH, "r", encoding="utf-8") as file:
            payload = json.load(file)

        df = geojson_to_df(payload)
        log.info("Extracted %d rows from the fallback file", len(df))
        return df


def read_stream_files() -> pd.DataFrame:
    """Read and combine JSON Lines files from the stream directory."""
    stream_dir = config.STREAM_DIR

    if not stream_dir.exists():
        log.info("Stream directory does not exist: %s", stream_dir)
        return pd.DataFrame()

    frames = []

    for path in sorted(stream_dir.glob("*.jsonl")):
        records = []

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    log.warning(
                        "Skipping invalid JSON in %s at line %d",
                        path.name,
                        line_number,
                    )

        if not records:
            continue

        # Handle files containing raw USGS GeoJSON features.
        if all("properties" in record for record in records):
            frame = geojson_to_df({"features": records})
        else:
            frame = pd.DataFrame(records)

        frames.append(frame)
        log.info("Read %d rows from %s", len(frame), path.name)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True, sort=False)
    log.info("Read %d total stream rows", len(result))
    return result


def transform(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean earthquake data and create model-ready features."""
    if raw.empty:
        log.warning("No raw data available to transform")
        return raw.copy()

    df = raw.copy()
    log.info("Starting transformation with %d rows", len(df))

    df = cleaning.epoch_ms_to_datetime(df)
    df = cleaning.dedupe_latest(df)
    log.info("After deduplication: %d rows", len(df))

    df = cleaning.keep_earthquakes(df)
    log.info("After keeping earthquakes: %d rows", len(df))

    df = cleaning.drop_missing_target(df, target="mag")
    log.info("After removing missing magnitudes: %d rows", len(df))

    df = features.add_time_features(df)
    df = features.add_quality_features(df)
    df = features.add_location_features(df)
    df = features.add_target(df)

    if "region" in df.columns:
        df["region"] = features.group_rare(df["region"], top_k=15)

    df = features.drop_leaky_columns(df)
    log.info("Transformation complete: %d rows, %d columns", *df.shape)

    return df


def load_sqlite(
    df: pd.DataFrame,
    db_path,
    table: str = "quakes",
) -> None:
    """Upsert records by id without creating duplicates."""
    if not table.replace("_", "").isalnum():
        raise ValueError("Invalid table name")

    if df.empty:
        return

    if "id" not in df.columns:
        raise ValueError("DataFrame must contain an 'id' column")

    data = df.copy()

    for column in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[column]):
            data[column] = data[column].astype(str)

    db_path = str(db_path)

    with sqlite3.connect(db_path) as connection:
        data.iloc[0:0].to_sql(
            table,
            connection,
            if_exists="append",
            index=False,
        )

        for record_id in data["id"].dropna().unique():
            connection.execute(
                f'DELETE FROM "{table}" WHERE id = ?',
                (record_id,),
            )

        data.to_sql(
            table,
            connection,
            if_exists="append",
            index=False,
        )

    log.info("Loaded %d rows into SQLite table '%s'", len(data), table)


def run_etl(mode: str = "batch") -> None:
    """Extract, optionally merge stream records, transform, and load."""
    if mode not in {"batch", "merge"}:
        raise ValueError("mode must be 'batch' or 'merge'")

    raw = extract()
    log.info("Extract stage: %d rows", len(raw))

    if mode == "merge":
        stream_df = read_stream_files()
        log.info("Stream stage: %d rows", len(stream_df))

        if not stream_df.empty:
            raw = pd.concat(
                [raw, stream_df],
                ignore_index=True,
                sort=False,
            )

        log.info("Merged raw data: %d rows", len(raw))

    processed = transform(raw)
    log.info("Transform stage: %d rows", len(processed))

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    load_sqlite(processed, DB_PATH)
    log.info("ETL completed successfully in %s mode", mode)


def main() -> None:
    """Configure logging and run the selected ETL mode."""
    parser = argparse.ArgumentParser(
        description="Run the earthquake ETL pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["batch", "merge"],
        default="batch",
        help="batch processes the USGS feed; merge includes stream files",
    )
    args = parser.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    run_etl(mode=args.mode)


if __name__ == "__main__":
    main()
