#!/usr/bin/env python3
# Generate schema-compatible placeholder portfolio weights and SHAP tables.
#
# These are deterministic placeholders generated with seed 42. They are not the
# final model outputs reported in the manuscript. They exist so that reviewers
# can verify column definitions, date coverage, and data types without
# accessing the proprietary reference implementation.

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

RNG = np.random.default_rng(42)

START = "2021-01-01"
END = "2025-12-31"


def main():
    feature_path = PROCESSED / "feature_matrix.parquet"
    if not feature_path.exists():
        raise FileNotFoundError("Missing feature matrix: {0}".format(feature_path))

    features = pd.read_parquet(feature_path)
    features["date"] = pd.to_datetime(features["date"])

    assets = sorted(features["asset"].unique())
    feature_names = [c for c in features.columns if c not in {"date", "asset"}]

    dates = pd.bdate_range(START, END, freq="B")

    # --- Placeholder portfolio weights ---
    weight_records = []
    for d in dates:
        w = RNG.dirichlet(np.ones(len(assets)))
        for asset, weight in zip(assets, w):
            weight_records.append(
                {"date": d, "asset": asset, "weight": float(weight)}
            )

    weights = pd.DataFrame(weight_records)
    weights.to_parquet(PROCESSED / "portfolio_weights.parquet", index=False)
    print("portfolio_weights.parquet -> {0}".format(weights.shape))

    # --- Placeholder predictive SHAP ---
    shap_pred_records = []
    for d in dates:
        for asset in assets:
            vals = RNG.normal(0.0, 0.01, len(feature_names))
            for fname, val in zip(feature_names, vals):
                shap_pred_records.append(
                    {"date": d, "asset": asset,
                     "feature": fname, "shap_value": float(val)}
                )

    shap_pred = pd.DataFrame(shap_pred_records)
    shap_pred.to_parquet(PROCESSED / "shap_predictive.parquet", index=False)
    print("shap_predictive.parquet -> {0}".format(shap_pred.shape))

    # --- Placeholder allocative SHAP ---
    shap_alloc_records = []
    for d in dates:
        for asset in assets:
            vals = RNG.normal(0.0, 0.005, len(feature_names))
            for fname, val in zip(feature_names, vals):
                shap_alloc_records.append(
                    {"date": d, "asset": asset,
                     "feature": fname, "shap_value": float(val)}
                )

    shap_alloc = pd.DataFrame(shap_alloc_records)
    shap_alloc.to_parquet(PROCESSED / "shap_allocative.parquet", index=False)
    print("shap_allocative.parquet -> {0}".format(shap_alloc.shape))


if __name__ == "__main__":
    main()