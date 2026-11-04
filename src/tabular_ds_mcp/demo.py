"""One-command demo: synthetic data, every tool in order, a short report.

    python -m tabular_ds_mcp.demo [folder]

Part 1 is a Kaggle-style card-fraud table (1.3% fraud, a deliberate shift in `amount` in the test set).
Part 2 shows why the validation scheme matters: a daily series whose level drifts over time, and customers
who appear in many rows. Random K-fold looks far better than it should on both; the tools say so.
"""
from __future__ import annotations

import argparse
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import core

WIDTH = 100


# ═══════════════════════════ synthetic data ═════════════════════════════════
def make_fraud(n: int, seed: int, amount_shift: float = 0.0) -> pd.DataFrame:
    """Card transactions with a rare fraud target, shaped like a Kaggle Playground table."""
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


def make_demo_data(folder: str = "demo_data", rows: int = 40_000) -> str:
    """Write train.csv, test.csv and sample_submission.csv (fraud data) and return the folder."""
    out = Path(folder).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    tr = make_fraud(rows, 1)
    te = make_fraud(rows // 2, 2, amount_shift=0.3).drop(columns="is_fraud")
    tr.insert(0, "id", np.arange(len(tr)))
    te.insert(0, "id", np.arange(len(tr), len(tr) + len(te)))
    tr.to_csv(out / "train.csv", index=False)
    te.to_csv(out / "test.csv", index=False)
    pd.DataFrame({"id": te["id"], "is_fraud": 0.0}).to_csv(out / "sample_submission.csv", index=False)
    return str(out)


def make_drifting_series(days: int = 1200, seed: int = 0) -> pd.DataFrame:
    """Two rows a day; the target follows a random walk, so neighbouring days are alike but the future is not
    predictable from the date."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=days, freq="D")
    level = np.cumsum(rng.normal(0, 1, days))
    day = np.repeat(np.arange(days), 2)
    df = pd.DataFrame({"id": np.arange(len(day)), "date": dates[day].strftime("%Y-%m-%d"),
                       "noise": rng.normal(size=len(day))})
    df["y"] = level[day] + rng.normal(0, 0.3, len(day))
    return df


def make_repeat_customers(n_customers: int = 400, per: int = 10, seed: int = 1) -> pd.DataFrame:
    """Ten rows per customer; the target depends on a hidden customer effect, and 'signup_score' identifies
    the customer without saying anything about new customers."""
    rng = np.random.default_rng(seed)
    effect, score = rng.normal(0, 2, n_customers), rng.normal(size=n_customers)
    cid = np.repeat(np.arange(n_customers), per)
    df = pd.DataFrame({"id": np.arange(len(cid)), "customer_id": cid, "signup_score": score[cid],
                       "x": rng.normal(size=len(cid))})
    df["y"] = (effect[cid] + 0.5 * df["x"] + rng.normal(0, 0.5, len(df)) > 1.5).astype(int)
    return df


# ═══════════════════════════ report ═════════════════════════════════════════
LABEL = 28


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width, break_on_hyphens=False) or [""]


def _line(label: str, text: str) -> None:
    lines = _wrap(text, WIDTH - LABEL)
    print(f"{label:<{LABEL}}{lines[0]}")
    for extra in lines[1:]:
        print(" " * LABEL + extra)


def _warn(text: str) -> None:
    for i, part in enumerate(_wrap(text, WIDTH - LABEL - 2)):
        print(" " * LABEL + ("! " if i == 0 else "  ") + part)


def run(folder: str = "tabular_ds_mcp_demo", rows: int = 40_000) -> dict:
    """Run every tool on synthetic data and print a short report. Returns the raw tool outputs."""
    t0 = time.time()
    root = Path(folder).expanduser()
    data = make_demo_data(str(root / "fraud"), rows)
    train, test, sample = (str(Path(data) / f) for f in ("train.csv", "test.csv", "sample_submission.csv"))
    out: dict = {}
    print(f"tabular-ds-mcp demo · synthetic card-fraud data in {data}\n")

    plan = out["inspect"] = core.inspect_competition(data)
    _line("[1/6] inspect_competition", f"target {plan['target']} · {plan['task']} · "
          f"{plan['submission_format']['kind']} submission · {plan['support']}")
    _line("", "metric hint: " + plan["metric_hint"].split(". Confirm")[0])

    prof = out["profile"] = core.profile_dataset(train, target="is_fraud", id_col="id")
    share = min(prof["target"]["class_share"].values())
    _line("[2/6] profile_dataset", f"{prof['rows']:,} rows · {prof['n_features']} features · "
          f"{100 * share:.2f}% fraud · {len(prof['warnings'])} warning(s)")
    for w in prof["warnings"]:
        _warn(w)

    shift = out["compare"] = core.compare_train_test(train, test, target="is_fraud", id_col="id")
    top = shift["features_by_psi"][0]
    _line("[3/6] compare_train_test", f"adversarial AUC {shift['adversarial_auc']:.2f} → {shift['shift_verdict']} · "
          f"most shifted: {top['feature']} (PSI {top['psi']:.3f})")

    cv = out["cv"] = core.cv_baseline(train, target="is_fraud", id_col="id", metric="roc_auc")
    s, b = cv["cv_scores"], cv["no_skill_baseline"]
    _line("[4/6] cv_baseline", f"ROC-AUC {s['roc_auc']:.3f} vs no-skill {b['roc_auc']:.3f} · "
          f"PR-AUC {s['pr_auc']:.3f} vs {b['pr_auc']:.3f} · fold SD {cv['fold_spread']['roc_auc_fold_std']:.3f}")

    ex = out["explain"] = core.explain_model(train, target="is_fraud", id_col="id")
    _line("[5/6] explain_model", "top: " + ", ".join(f["feature"] for f in ex["top_features"][:3]) +
          " · no gain: " + (", ".join(ex["features_with_no_gain"]) or "none"))

    sub = str(root / "submission.csv")
    core.fit_predict(train, test, target="is_fraud", id_col="id", out_path=sub, sample_submission_path=sample)
    chk = out["check"] = core.check_submission(sub, sample, test_path=test)
    passed = sum(c["ok"] for c in chk["checks"])
    _line("[6/6] fit_predict + check", f"submission.csv written · {passed}/{len(chk['checks'])} checks pass · "
          f"ready to upload: {chk['ok']}")

    print("\nWhy the validation scheme matters")
    series = root / "drifting_series.csv"
    make_drifting_series().to_csv(series, index=False)
    ts = out["time"] = core.cv_baseline(str(series), target="y", id_col="id", time_col="date", metric="rmse")
    c = ts["random_split_comparison"]
    _line("time_col='date'", f"RMSE {c['random_split']:.2f} with random splits, {c['time-ordered']:.2f} with "
          "time-ordered folds")
    _line("", f"→ {c['verdict']}")
    customers = root / "repeat_customers.csv"
    make_repeat_customers().to_csv(customers, index=False)
    gr = out["group"] = core.cv_baseline(str(customers), target="y", id_col="id", group_col="customer_id",
                                         metric="roc_auc")
    c = gr["random_split_comparison"]
    _line("group_col='customer_id'", f"ROC-AUC {c['random_split']:.2f} with random splits, {c['group']:.2f} "
          "on unseen customers")
    _line("", f"→ {c['verdict']}")
    print(f"\ndone in {time.time() - t0:.0f} s · files in {root}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", nargs="?", default="tabular_ds_mcp_demo", help="where to write the demo files")
    run(ap.parse_args().folder)


if __name__ == "__main__":
    main()
