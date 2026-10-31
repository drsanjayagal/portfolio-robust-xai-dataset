#!/usr/bin/env python3
# Download price panels via yfinance.

from pathlib import Path

import yfinance as yf

TICKERS = {
    "sp500_proxy": ["SPY", "QQQ", "IWM", "DIA", "VTI"],
    "etf": ["SPY", "EFA", "AGG", "VNQ", "GLD", "TLT"],
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"],
}

START_DATE = "2005-01-01"
END_DATE = "2025-12-31"

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "prices"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def flatten_columns(df):
    if getattr(df.columns, "nlevels", 1) > 1:
        df.columns = [
            "_".join(str(level) for level in col if level)
            for col in df.columns
        ]
    return df


def main():
    for label, tickers in TICKERS.items():
        out_path = RAW_DIR / (label + ".parquet")
        if out_path.exists():
            print("{0} already exists; skipping.".format(out_path.name))
            continue

        print("Downloading {0}: {1}".format(label, tickers))
        df = yf.download(
            tickers=tickers,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
        df = flatten_columns(df)
        df.index.name = "date"
        df.to_parquet(out_path)
        print("  saved {0} ({1} rows, {2} cols)".format(
            out_path, df.shape[0], df.shape[1]))

    print("Price download complete.")


if __name__ == "__main__":
    main()