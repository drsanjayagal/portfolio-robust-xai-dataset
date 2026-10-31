#!/usr/bin/env python3
# Validate the processed dataset schema and basic invariants.
#
# This script tolerates both storage conventions for processed tables:
#   - 'date' stored as a normal column
#   - 'date' stored as the DataFrame index
# so that validation does not fail purely because of a save/load detail.

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

REQUIRED_FILES = [
    "feature_matrix.parquet",
    "regime_probabilities.parquet",
    "portfolio_weights.parquet",
    "shap_predictive.parquet",
    "shap_allocative.parquet",
]


def normalize_date(df):
    # Ensure that a 'date' column exists, whether it was saved as a column or
    # as the DataFrame index.
    if "date" in df.columns:
        return df
    if df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
        return df.reset_index().rename(columns={"index": "date"})
    return df


def main():
    for name in REQUIRED_FILES:
        path = PROCESSED / name
        if not path.exists():
            raise FileNotFoundError("Missing processed file: {0}".format(path))
        print("OK: {0}".format(name))

    # --- feature_matrix ---
    features = normalize_date(pd.read_parquet(PROCESSED / "feature_matrix.parquet"))
    required_feature_cols = {
        "date", "asset", "vol_5d", "vol_21d", "vol_63d",
        "downside_dev_21d", "downside_dev_63d", "skew_63d", "kurt_63d",
        "mom_21d", "mom_63d", "mom_252d", "corr_mkt_63d",
        "vix", "term_spread", "credit_spread", "nfci", "pred_uncertainty",
    }
    missing = required_feature_cols - set(features.columns)
    if missing:
        raise ValueError("feature_matrix missing columns: {0}".format(missing))
    print("OK: feature_matrix columns ({0} rows)".format(features.shape[0]))

    # --- regime_probabilities ---
    regime = normalize_date(pd.read_parquet(PROCESSED / "regime_probabilities.parquet"))
    regime_cols = {"date", "regime_low", "regime_medium", "regime_high"}
    if not regime_cols.issubset(regime.columns):
        raise ValueError(
            "regime_probabilities missing required columns. Got: {0}".format(
                list(regime.columns)))
    sums = regime[["regime_low", "regime_medium", "regime_high"]].sum(axis=1)
    if not ((sums - 1.0).abs() < 1e-6).all():
        raise ValueError("Regime probabilities do not sum to 1.")
    print("OK: regime_probabilities ({0} rows)".format(regime.shape[0]))

    # --- portfolio_weights ---
    weights = normalize_date(pd.read_parquet(PROCESSED / "portfolio_weights.parquet"))
    if not {"date", "asset", "weight"}.issubset(weights.columns):
        raise ValueError(
            "portfolio_weights missing required columns. Got: {0}".format(
                list(weights.columns)))
    weight_sums = weights.groupby("date")["weight"].sum()
    if not ((weight_sums - 1.0).abs() < 1e-6).all():
        raise ValueError("Portfolio weights do not sum to 1 per date.")
    print("OK: portfolio_weights ({0} rows)".format(weights.shape[0]))

    # --- SHAP tables ---
    for name in ["shap_predictive.parquet", "shap_allocative.parquet"]:
        df = normalize_date(pd.read_parquet(PROCESSED / name))
        required = {"date", "asset", "feature", "shap_value"}
        if not required.issubset(df.columns):
            raise ValueError("{0} missing required columns. Got: {1}".format(
                name, list(df.columns)))
        print("OK: {0} ({1} rows)".format(name, df.shape[0]))

    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()