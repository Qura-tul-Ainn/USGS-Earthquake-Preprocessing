
"""Task 2: simulate streaming by polling a feed on a timer."""

import argparse
import time
from datetime import datetime

import pandas as pd

from quakes.fetch import fetch_feed, geojson_to_df


def filter_unseen(df: pd.DataFrame, seen: set) -> pd.DataFrame:
    """Return unseen (id, updated) pairs and add them to seen."""
    """This function filters out previously seen earthquake records and returns only new or updated records"""
    if df.empty:
        return df.copy()

    mask = []

    for event_id, updated in zip(df["id"], df["updated"]):
        # Convert NumPy integer values to Python integers when needed.
        if pd.notna(updated):
            updated = int(updated) if isinstance(updated, (int, float)) else updated

        pair = (event_id, updated) #on the basis of event_id and updated timestamp earthquake records is seen before or not
        is_new = pair not in seen
        mask.append(is_new)

        if is_new:
            seen.add(pair)

    return df.loc[mask].copy()  #return new records

"""the main function that runs the entire streaming process. The feed parameter specifies which USGS feed to use, interval_s defines how many seconds to wait before fetching the data again, duration_s specifies how long the streaming process should run, and out_path defines the file where the data will be saved"""


def run_stream(
    feed: str,
    interval_s: int,
    duration_s: int,
    out_path: str,
) -> None:
    """Poll a feed and append unseen events to a JSON Lines file."""
    from pathlib import Path
    #interval and duration must be positive, otherwise raise a ValueError
    if interval_s <= 0:
        raise ValueError("interval_s must be greater than zero.")
    if duration_s < 0:
        raise ValueError("duration_s cannot be negative.")
    # make sure the output directory exists
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    start_time = time.monotonic()

#fetch data according to the specified interval and duration
    while time.monotonic() - start_time < duration_s:
        try:
            payload = fetch_feed(feed)
            df = geojson_to_df(payload)
            new_rows = filter_unseen(df, seen)
            #fetch convert updte 

            if not new_rows.empty:
                new_rows.to_json(
                    output_path,
                    orient="records", #each row is a record
                    lines=True, # each record is on a new line
                    mode="a", # append new data without overwriting existing data
                    date_format="iso",#date format in ISO standard 
                )

            timestamp = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{timestamp}] "
                f"{len(new_rows)} new / {len(df)} fetched"
            ) #print streaming report

               #error handling 
            """catches errors during polling and displays the timestamp and error message instead of hiding the problem"""
            #[2026-10-25 10:00:00] Poll failed: Connection timed out: for example
        except Exception as exc:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] Poll failed: {exc}")

        """This statement pauses the program for the smaller of the polling interval and the remaining streaming time"""
        remaining = duration_s - (time.monotonic() - start_time)
        if remaining > 0:
            time.sleep(min(interval_s, remaining))
        """Example: If interval_s = 10 seconds and remaining = 4 seconds, the program will wait for only 4 seconds to ensure that the total streaming duration is not exceeded"""

def main() -> None:
    """Parse command-line arguments and start streaming."""
    parser = argparse.ArgumentParser(
        description="Poll a USGS earthquake feed and save new events."
    )
    parser.add_argument("--feed", default="all_hour") #last hour feed
    parser.add_argument("--interval", type=int, default=60) #Every 60 seconds poll the feed for new earthquake events
    parser.add_argument("--duration", type=int, default=2400) # total duration of 2400 seconds (40 minutes) 
    parser.add_argument("--out", default="data/stream/stream.jsonl")# save data in file named stream.jsonl in data/stream 

    args = parser.parse_args()

    run_stream(
        feed=args.feed,
        interval_s=args.interval,
        duration_s=args.duration,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
