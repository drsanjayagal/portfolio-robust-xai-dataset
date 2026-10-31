#!/usr/bin/env python3
"""
One-command build for the portfolio-robust-xai-dataset repository.
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "01_download_ff_factors.py",
    "02_download_fred.py",
    "03_download_prices.py",
    "04_build_features.py",
    "05_fit_regime_model.py",
    "06_build_placeholder_outputs.py",
    "07_validate_dataset.py",
]


def main() -> int:
    script_dir = Path(__file__).resolve().parent / "scripts"
    for name in SCRIPTS:
        script = script_dir / name
        if not script.exists():
            print(f"Missing script: {script}", file=sys.stderr)
            return 1
        print(f"\n{'=' * 70}\nRunning {name}\n{'=' * 70}")
        result = subprocess.run([sys.executable, str(script)])
        if result.returncode != 0:
            print(f"Script {name} failed with exit code {result.returncode}", file=sys.stderr)
            return result.returncode
    print("\nDataset build complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())