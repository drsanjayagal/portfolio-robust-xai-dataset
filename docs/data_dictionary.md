# Data Dictionary

## `data/processed/feature_matrix.parquet`

Long-format table with one row per `(date, asset)`.

| Column | Description |
|---|---|
| `date` | Trading date |
| `asset` | Asset identifier |
| `vol_5d` | Rolling 5-day volatility of daily returns |
| `vol_21d` | Rolling 21-day volatility |
| `vol_63d` | Rolling 63-day volatility |
| `downside_dev_21d` | Downside deviation over 21 days |
| `downside_dev_63d` | Downside deviation over 63 days |
| `skew_63d` | Skewness of returns over 63 days |
| `kurt_63d` | Excess kurtosis over 63 days |
| `mom_21d` | 21-day cumulative momentum |
| `mom_63d` | 63-day cumulative momentum |
| `mom_252d` | 252-day cumulative momentum |
| `corr_mkt_63d` | Realized correlation with the equal-weight market proxy over 63 days |
| `vix` | CBOE Volatility Index |
| `term_spread` | 10-year minus 3-month Treasury spread |
| `credit_spread` | Baa minus 10-year Treasury spread |
| `nfci` | Chicago Fed National Financial Conditions Index |
| `pred_uncertainty` | Deterministic uncertainty proxy based on rolling return dispersion |

## `data/processed/regime_probabilities.parquet`

Wide-format table with one row per trading date.

| Column | Description |
|---|---|
| `date` | Trading date |
| `regime_low` | Posterior probability of the low-volatility regime |
| `regime_medium` | Posterior probability of the medium-volatility regime |
| `regime_high` | Posterior probability of the high-volatility regime |

Each row sums to 1.

## `data/processed/portfolio_weights.parquet`

Long-format table with one row per `(date, asset)`.

| Column | Description |
|---|---|
| `date` | Rebalancing date |
| `asset` | Asset identifier |
| `weight` | Portfolio weight |

For each date, weights are long-only and sum to 1.

## `data/processed/shap_predictive.parquet`

Long-format table with one row per `(date, asset, feature)`.

| Column | Description |
|---|---|
| `date` | Trading date |
| `asset` | Asset identifier |
| `feature` | Feature name |
| `shap_value` | Predictive-level SHAP value |

## `data/processed/shap_allocative.parquet`

Long-format table with one row per `(date, asset, feature)`.

| Column | Description |
|---|---|
| `date` | Trading date |
| `asset` | Asset identifier |
| `feature` | Feature name |
| `shap_value` | Allocation-level SHAP value |

## Placeholder notice

`portfolio_weights.parquet`, `shap_predictive.parquet`, and `shap_allocative.parquet` are deterministic schema-compatible placeholders generated with seed `42`. They are not the final model outputs reported in the manuscript. The proprietary reference implementation produces the final numerical values and is available to qualified reviewers upon request.