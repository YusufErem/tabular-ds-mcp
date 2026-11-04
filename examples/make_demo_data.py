"""Write the synthetic card-fraud demo tables (train.csv, test.csv, sample_submission.csv).
The test set has a deliberate shift in `amount`, so compare_train_test has something to find.

Usage: python examples/make_demo_data.py [--rows 40000] [--out demo_data]
(The full walkthrough runs with: python -m tabular_ds_mcp.demo)
"""
import argparse
from pathlib import Path

from tabular_ds_mcp.demo import make_demo_data

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=40_000)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "demo_data"))
    a = ap.parse_args()
    print("wrote", make_demo_data(a.out, a.rows))
