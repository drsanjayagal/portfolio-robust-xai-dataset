# Portfolio Robust XAI Dataset

Reproducible dataset and data-generation pipeline for the manuscript:

**A Machine Learning Framework for Robust and Explainable Portfolio Optimization Under Market Uncertainty**

Repository: `https://github.com/drsanjayagal/portfolio-robust-xai-dataset`

## What is included

- Public scripts that download the third-party raw data from their official providers.
- A deterministic pipeline that constructs the processed artefacts described in the manuscript.
- Processed artefacts committed under `data/processed/` for direct inspection and verification.
- A validation script that checks schema, date coverage, weight sums, and regime-probability sums.

## What is not included

The proprietary reference implementation of the model (heteroscedastic LSTM, deep ensemble, regime-conditioned distributionally robust optimizer, and full KernelSHAP computation) is **not** included in this repository. It is available from the corresponding author upon reasonable written request under a standard non-disclosure agreement, as stated in the manuscript.

The `portfolio_weights.parquet`, `shap_predictive.parquet`, and `shap_allocative.parquet` files in `data/processed/` are schema-compatible placeholders generated deterministically with seed `42`. They allow reviewers to verify column definitions, date coverage, and data types. The manuscript reports the actual numerical values produced by the proprietary reference implementation.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python build_dataset.py

This runs, in order:

Fama–French daily factor download

FRED macroeconomic series download

Yahoo Finance price panel download via yfinance

Feature construction

Three-state Gaussian HMM regime fitting

Placeholder weight and SHAP generation

Dataset validation

Data sources
Raw data are obtained at runtime from the Kenneth French Data Library, the Federal Reserve Economic Data repository, and Yahoo Finance via yfinance. These raw files are not redistributed in this repository where the providers' terms do not permit redistribution.

Licence
The processed derived data in data/processed/ are released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0). Raw third-party data remain subject to the terms of their respective providers.

Contact
Corresponding author: Sanjay Agal
sanjay.agal32685@paruluniversity.ac.in