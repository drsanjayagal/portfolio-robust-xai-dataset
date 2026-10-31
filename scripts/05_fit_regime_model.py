#!/usr/bin/env python3
# Fit a three-state Gaussian hidden Markov model and save regime probabilities.

from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main():
    path = PROCESSED / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError("Missing feature matrix: {0}".format(path))

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])

    agg = (
        df.groupby("date")[["vol_21d", "corr_mkt_63d", "mom_63d"]]
        .mean()
        .sort_index()
        .dropna()
    )

    if len(agg) < 10:
        raise ValueError("Too few observations to fit the regime model.")

    X = StandardScaler().fit_transform(agg.values)

    model = GaussianHMM(
        n_components=3,
        covariance_type="diag",
        n_iter=200,
        random_state=42,
    )
    model.fit(X)
    probs = model.predict_proba(X)

    # Order states by average volatility so that regime_low has the lowest vol.
    state_vol = []
    for k in range(3):
        mask = probs.argmax(axis=1) == k
        if mask.any():
            state_vol.append(agg["vol_21d"].values[mask].mean())
        else:
            state_vol.append(np.inf)
    order = np.argsort(state_vol)
    probs = probs[:, order]

    out = pd.DataFrame(
        probs,
        columns=["regime_low", "regime_medium", "regime_high"],
    )
    out.insert(0, "date", agg.index)

    out.to_parquet(PROCESSED / "regime_probabilities.parquet", index=False)
    print("regime_probabilities.parquet -> {0}".format(out.shape))


if __name__ == "__main__":
    main()