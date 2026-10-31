#!/usr/bin/env python3
# Download Fama-French daily 5-factor and momentum data from the official
# Kenneth French Data Library.
#
# Corrected URLs use lowercase 'ken.french' and the official ftp/ CSV archives.
#
# The FF5 and Momentum files have DIFFERENT numbers of preamble lines, so this
# parser does NOT use a fixed skiprows value. Instead it:
#   1. Locates the header row (first line whose first cell is empty).
#   2. Collects only rows whose first cell is an 8-digit YYYYMMDD date.
#   3. Stops at the first non-date content line (e.g. 'Annual Factors').

from pathlib import Path
import io
import zipfile

import pandas as pd
import requests

FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Momentum_Factor_daily_CSV.zip"
)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "fama_french"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_zip(url, dest_dir, timeout=60):
    print("Fetching " + url)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(dest_dir)
        extracted = [dest_dir / name for name in zf.namelist()]
    print("  extracted {0} file(s) to {1}".format(len(extracted), dest_dir))
    return extracted


def parse_ff_csv_to_parquet(csv_path, out_path):
    # Read the raw file line by line.
    with open(csv_path, "r", encoding="latin-1") as f:
        lines = f.readlines()

    # ---- 1. Locate header row ---------------------------------------
    # Kenneth French CSV header rows start with an empty first cell,
    # e.g. ',Mkt-RF,SMB,HML,RMW,CMA,RF' or ',Mom'.
    header_idx = None
    for i, line in enumerate(lines):
        parts = line.rstrip("\n").split(",")
        if len(parts) >= 2 and parts[0].strip() == "":
            # Require at least one alphabetic column name to avoid
            # matching a description line that happens to start with ','.
            if any(any(ch.isalpha() for ch in p) for p in parts[1:]):
                header_idx = i
                break

    if header_idx is None:
        raise ValueError(
            "Could not locate header row in {0}. "
            "First 5 lines were:\n{1}".format(
                csv_path,
                "".join(lines[:5]),
            )
        )

    header = [p.strip() for p in lines[header_idx].rstrip("\n").split(",")]
    header[0] = "date"
    n_cols = len(header)

    # ---- 2. Collect only daily data rows ----------------------------
    data_rows = []
    for line in lines[header_idx + 1:]:
        stripped = line.rstrip("\n")
        if not stripped:
            # Blank line: skip and keep scanning.
            continue

        parts = stripped.split(",")
        if len(parts) < 2:
            # Section labels such as 'Annual Factors: January-December'.
            continue

        first = parts[0].strip()
        if len(first) == 8 and first.isdigit():
            # Pad short rows with empty strings so the DataFrame stays square.
            if len(parts) < n_cols:
                parts = parts + [""] * (n_cols - len(parts))
            data_rows.append(parts[:n_cols])
        else:
            # First non-date content line: the daily section has ended.
            break

    if not data_rows:
        raise ValueError("No daily data rows found in {0}".format(csv_path))

    # ---- 3. Build DataFrame -----------------------------------------
    df = pd.DataFrame(data_rows, columns=header)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0

    df.to_parquet(out_path)
    print("  parsed {0} -> {1} ({2} rows, {3} cols)".format(
        csv_path.name, out_path, df.shape[0], df.shape[1]))


def main():
    ff5_dir = RAW_DIR / "ff5"
    for f in fetch_zip(FF5_URL, ff5_dir):
        if f.suffix.lower() == ".csv":
            parse_ff_csv_to_parquet(f, RAW_DIR / "ff5_daily.parquet")
            break

    mom_dir = RAW_DIR / "mom"
    for f in fetch_zip(MOM_URL, mom_dir):
        if f.suffix.lower() == ".csv":
            parse_ff_csv_to_parquet(f, RAW_DIR / "mom_daily.parquet")
            break

    print("Fama-French download complete.")


if __name__ == "__main__":
    main()