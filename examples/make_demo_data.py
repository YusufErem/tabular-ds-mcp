"""Synthetic card-transaction data with a rare fraud target (about 1.7%), shaped like a Kaggle Playground table.
Writes demo_data/train.csv, test.csv and sample_submission.csv. The test set has a deliberate shift in `amount`
so that compare_train_test has something to find.

Usage: python examples/make_demo_data.py [--rows 40000]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make(n: int, seed: int, amount_shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, n)
    amount = np.round(rng.lognormal(3.6 + amount_shift, 1.0, n), 2)
    merchant = rng.choice(["grocery", "fuel", "online", "travel", "electronics", "restaurant"], n,
                          p=[0.3, 0.15, 0.25, 0.08, 0.07, 0.15])
    distance_km = np.round(rng.exponential(12, n), 1)
    card_age_days = rng.integers(1, 3000, n)
    tx_last_24h = rng.poisson(2, n)
    foreign = rng.random(n) < 0.06
    logit = (-5.6 + 0.9 * (merchant == "online") + 1.1 * (merchant == "electronics") + 1.3 * foreign
             + 0.35 * np.log1p(amount) * (hour < 6) + 0.25 * tx_last_24h - 0.0004 * card_age_days
             + 0.01 * distance_km)
    is_fraud = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"hour": hour, "amount": amount, "merchant": merchant, "distance_km": distance_km,
                       "card_age_days": card_age_days, "tx_last_24h": tx_last_24h, "foreign": foreign,
                       "is_fraud": is_fraud})
    df.loc[rng.random(n) < 0.03, "distance_km"] = np.nan          # some missing values
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=40_000)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "demo_data"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    tr = make(a.rows, 1)
    te = make(a.rows // 2, 2, amount_shift=0.3).drop(columns="is_fraud")
    tr.insert(0, "id", np.arange(len(tr)))
    te.insert(0, "id", np.arange(len(tr), len(tr) + len(te)))
    tr.to_csv(out / "train.csv", index=False)
    te.to_csv(out / "test.csv", index=False)
    pd.DataFrame({"id": te["id"], "is_fraud": 0.0}).to_csv(out / "sample_submission.csv", index=False)
    print(f"wrote {out} | fraud rate {tr.is_fraud.mean():.4f}")
