# Reproducibility Notes

## Environment

- Python 3.10+
- Dependencies listed in `requirements.txt`
- GPU not required for dataset construction; the proprietary model implementation separately requires GPU acceleration for deep ensemble training.

## Command

```bash
python build_dataset.py
python scripts/01_download_ff_factors.py
python scripts/02_download_fred.py
python scripts/03_download_prices.py
python scripts/04_build_features.py
python scripts/05_fit_regime_model.py
python scripts/06_build_placeholder_outputs.py
python scripts/07_validate_dataset.py

Data coverage
Raw prices: 2005-01-01 to 2025-12-31, subject to provider availability.

Processed feature matrix: dates for which sufficient rolling history is available.

Regime probabilities: same date coverage as the processed feature matrix after aggregation.

Placeholder weights and SHAP outputs: business-day calendar from 2021-01-01 to 2025-12-31.

Validation
scripts/07_validate_dataset.py checks:

presence of all processed files;

required columns;

no missing dates in regime probabilities;

weights sum to 1 per date;

regime probabilities sum to 1 per date;

non-empty SHAP tables.