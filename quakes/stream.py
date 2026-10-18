
"""Task 2: simulate streaming by polling a feed on a timer."""

import argparse
import time
from datetime import datetime

import pandas as pd

from quakes.fetch import fetch_feed, geojson_to_df


def filter_unseen(df: pd.DataFrame, seen: set) -> pd.DataFrame:
    """Return unseen (id, updated) pairs and add them to seen."""
    if df.empty:
        return df.copy()

    mask = []

    for event_id, updated in zip(df["id"], df["updated"]):
        # Convert NumPy integer values to Python integers when needed.
        if pd.notna(updated):
            updated = int(updated) if isinstance(updated, (int, float)) else updated

        pair = (event_id, updated)
        is_new = pair not in seen
        mask.append(is_new)

        if is_new:
            seen.add(pair)

    return df.loc[mask].copy()


def run_stream(
    feed: str,
    interval_s: int,
    duration_s: int,
    out_path: str,
) -> None:
    """Poll a feed and append unseen events to a JSON Lines file."""
    from pathlib import Path

    if interval_s <= 0:
        raise ValueError("interval_s must be greater than zero.")
    if duration_s < 0:
        raise ValueError("duration_s cannot be negative.")

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    start_time = time.monotonic()

    while time.monotonic() - start_time < duration_s:
        try:
            payload = fetch_feed(feed)
            df = geojson_to_df(payload)
            new_rows = filter_unseen(df, seen)

            if not new_rows.empty:
                new_rows.to_json(
                    output_path,
                    orient="records",
                    lines=True,
                    mode="a",
                    date_format="iso",
                )

            timestamp = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{timestamp}] "
                f"{len(new_rows)} new / {len(df)} fetched"
            )

        except Exception as exc:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] Poll failed: {exc}")

        remaining = duration_s - (time.monotonic() - start_time)
        if remaining > 0:
            time.sleep(min(interval_s, remaining))


def main() -> None:
    """Parse command-line arguments and start streaming."""
    parser = argparse.ArgumentParser(
        description="Poll a USGS earthquake feed and save new events."
    )
    parser.add_argument("--feed", default="all_hour")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--duration", type=int, default=2400)
    parser.add_argument("--out", default="data/stream/stream.jsonl")

    args = parser.parse_args()

    run_stream(
        feed=args.feed,
        interval_s=args.interval,
        duration_s=args.duration,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
