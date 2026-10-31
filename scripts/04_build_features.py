#!/usr/bin/env python3
# Build the long-format feature matrix from raw price and macro data.

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def load_close_prices():
    path = RAW / "prices" / "sp500_proxy.parquet"
    if not path.exists():
        raise FileNotFoundError("Missing price file: {0}".format(path))

    df = pd.read_parquet(path)
    close_cols = [
        c for c in df.columns
        if c.lower().endswith("_close") or c.lower() == "close"
    ]
    if not close_cols:
        raise ValueError("No close-price columns found in price panel.")

    out = {}
    for c in close_cols:
        if c.lower() == "close":
            ticker = "ASSET"
        else:
            ticker = c.rsplit("_", 1)[0]
        out[ticker] = pd.to_numeric(df[c], errors="coerce")

    close = pd.DataFrame(out)
    close.index = pd.to_datetime(df.index)
    close.index.name = "date"
    close = close.sort_index().dropna(how="all")
    return close


def load_macro(dates):
    series = {}
    for name in ["VIXCLS", "T10Y3M", "BAA10Y", "NFCI"]:
        path = RAW / "fred" / (name + ".parquet")
        if not path.exists():
            continue
        s = pd.read_parquet(path).iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        s.index.name = "date"
        series[name] = s.sort_index()

    macro = pd.DataFrame(series)
    macro = macro.reindex(dates).ffill().bfill()
    macro = macro.rename(
        columns={
            "VIXCLS": "vix",
            "T10Y3M": "term_spread",
            "BAA10Y": "credit_spread",
            "NFCI": "nfci",
        }
    )
    macro.index.name = "date"
    return macro


def build_asset_features(returns):
    records = []
    market = returns.mean(axis=1)

    for ticker in returns.columns:
        r = returns[ticker].dropna()
        if len(r) < 30:
            continue

        df = pd.DataFrame(index=r.index)
        df["asset"] = ticker

        for w in [5, 21, 63]:
            df["vol_{0}d".format(w)] = r.rolling(w).std()

        for w in [21, 63]:
            neg = r.where(r < 0.0, 0.0)
            df["downside_dev_{0}d".format(w)] = neg.rolling(w).std()

        df["skew_63d"] = r.rolling(63).skew()
        df["kurt_63d"] = r.rolling(63).kurt()

        for w in [21, 63, 252]:
            df["mom_{0}d".format(w)] = (
                (1.0 + r).rolling(w).apply(np.prod, raw=True) - 1.0
            )

        df["corr_mkt_63d"] = r.rolling(63).corr(market)

        df["pred_uncertainty"] = r.rolling(21).std()

        df = df.reset_index().rename(columns={"index": "date"})
        records.append(df)

    if not records:
        raise ValueError("No asset features could be constructed.")

    long = pd.concat(records, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"])
    return long


def main():
    close = load_close_prices()
    returns = close.pct_change()

    features = build_asset_features(returns)
    dates = pd.DatetimeIndex(sorted(features["date"].unique()))
    macro = load_macro(dates)

    macro_reset = macro.reset_index()
    if "date" not in macro_reset.columns and "index" in macro_reset.columns:
        macro_reset = macro_reset.rename(columns={"index": "date"})

    features = features.merge(macro_reset, on="date", how="left")
    features = features.sort_values(["date", "asset"]).reset_index(drop=True)

    out_path = PROCESSED / "feature_matrix.parquet"
    features.to_parquet(out_path, index=False)
    print("feature_matrix.parquet -> {0}".format(features.shape))
    print("  date range: {0} to {1}".format(
        features["date"].min(), features["date"].max()))
    print("  assets: {0}".format(features["asset"].nunique()))


if __name__ == "__main__":
    main()