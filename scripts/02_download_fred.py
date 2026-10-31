#!/usr/bin/env python3
# Download FRED macroeconomic series used in the manuscript.
#
# FRED's fredgraph.csv endpoint may return the date column as either
# "observation_date" (current default) or "DATE" (older API versions).
# This script detects the date column dynamically instead of hard-coding it.

from pathlib import Path
import io

import pandas as pd
import requests

FRED_SERIES = {
    "VIXCLS": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS",
    "T10Y3M": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y3M",
    "BAA10Y": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAA10Y",
    "NFCI": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=NFCI",
}

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "fred"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def detect_date_column(df):
    # Return the first column whose name contains "date" (case-insensitive).
    for col in df.columns:
        if "date" in col.lower():
            return col
    raise ValueError(
        "No date column found. Columns were: {0}".format(list(df.columns))
    )


def fetch_series(series_id, url, timeout=60):
    print("Fetching " + series_id)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))

    date_col = detect_date_column(df)

    # FRED uses "." for missing observations.
    df = df.replace(".", pd.NA)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col)
    df.index.name = "date"

    # Coerce the series column to numeric.
    df = df.apply(pd.to_numeric, errors="coerce")

    df.to_csv(RAW_DIR / (series_id + ".csv"))
    df.to_parquet(RAW_DIR / (series_id + ".parquet"))
    print("  saved {0}: {1} observations (date column: {2})".format(
        series_id, df.shape[0], date_col))


def main():
    for series_id, url in FRED_SERIES.items():
        fetch_series(series_id, url)
    print("FRED download complete.")


if __name__ == "__main__":
    main()