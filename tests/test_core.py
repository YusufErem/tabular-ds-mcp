"""Evaluation harness: every tool is checked on small synthetic tables whose right answer is known."""
import numpy as np
import pandas as pd
import pytest

import tabular_ds_mcp as t
from tabular_ds_mcp.core import _psi


def _csv(tmp_path, name, df):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def binary(tmp_path):
    """x1 drives the target strongly, noise does not; about 3.5% positives."""
    rng = np.random.default_rng(0)
    n = 4000
    x1 = rng.normal(size=n)
    df = pd.DataFrame({"id": np.arange(n), "x1": x1, "noise": rng.normal(size=n),
                       "cat": rng.choice(["a", "b", "c"], n)})
    df["y"] = (x1 + 0.3 * rng.normal(size=n) > 1.9).astype(int)
    return df


# ── profile ───────────────────────────────────────────────────────────────
def test_profile_flags_known_problems(tmp_path, binary):
    df = binary.copy()
    df["const"] = 1
    df.loc[:399, "noise"] = np.nan                                    # exactly 10% missing
    df["leak"] = df["y"] + np.random.default_rng(1).normal(0, 0.01, len(df))
    df = pd.concat([df, df.iloc[:7].assign(id=-np.arange(1, 8))])     # 7 duplicate rows, new IDs
    r = t.profile_dataset(_csv(tmp_path, "tr.csv", df), target="y", id_col="id")
    w = " ".join(r["warnings"])
    assert "'const' is constant" in w
    assert "imbalanced" in w
    assert "Possible target leakage" in w and "leak" in w
    assert r["duplicate_rows"] == 7
    noise = next(c for c in r["column_summary"] if c["column"] == "noise")
    # 400 missing + the 7 copied rows, which come from the missing block
    assert noise["missing_pct"] == pytest.approx(100 * 407 / len(df), abs=1e-6)
    assert r["target"]["task"] == "binary"


def test_profile_detects_id_like_column(tmp_path, binary):
    r = t.profile_dataset(_csv(tmp_path, "tr.csv", binary.rename(columns={"id": "row_key"})), target="y")
    assert any("row_key" in w and "ID" in w for w in r["warnings"])


# ── PSI and train/test comparison ─────────────────────────────────────────
def test_psi_is_small_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=20000))
    assert _psi(a, pd.Series(rng.normal(size=20000))) < 0.02
    assert _psi(a, pd.Series(rng.normal(1.0, 1, 20000))) > 0.25


def test_psi_keeps_missing_values_in_their_own_bin():
    a = pd.Series(["x"] * 50 + ["y"] * 50)
    b = pd.Series(["x"] * 50 + [None] * 50)                           # half the test values missing
    assert _psi(a, b) > 1.0


def test_compare_same_distribution(tmp_path, binary):
    tr, te = binary.iloc[:2000], binary.iloc[2000:].drop(columns="y")
    r = t.compare_train_test(_csv(tmp_path, "tr.csv", tr), _csv(tmp_path, "te.csv", te), target="y", id_col="id")
    assert r["adversarial_auc"] < 0.56 and r["shift_verdict"] == "no detectable shift"


def test_compare_detects_shift_mismatch_and_unseen_categories(tmp_path, binary):
    tr = binary.iloc[:2000]
    te = binary.iloc[2000:].drop(columns="y").copy()
    te["x1"] += 1.5
    te.loc[te.index[:50], "cat"] = "z"                                 # category never seen in train
    te["extra"] = 0
    r = t.compare_train_test(_csv(tmp_path, "tr.csv", tr), _csv(tmp_path, "te.csv", te), target="y", id_col="id")
    assert r["adversarial_auc"] > 0.65 and r["shift_verdict"] == "strong shift"
    assert r["features_by_psi"][0]["feature"] == "x1"
    w = " ".join(r["warnings"])
    assert "Column mismatch" in w and "extra" in w and "'cat'" in w


# ── cross-validated baseline ──────────────────────────────────────────────
def test_cv_beats_no_skill_on_signal(tmp_path, binary):
    oof = tmp_path / "oof.csv"
    r = t.cv_baseline(_csv(tmp_path, "tr.csv", binary), target="y", id_col="id", oof_path=str(oof))
    assert r["task"] == "binary"
    assert r["cv_scores"]["roc_auc"] > 0.95 > r["no_skill_baseline"]["roc_auc"]
    assert r["cv_scores"]["pr_auc"] > 5 * r["no_skill_baseline"]["pr_auc"]
    assert len(pd.read_csv(oof)) == len(binary)


def test_cv_is_near_chance_on_pure_noise(tmp_path, binary):
    df = binary.copy()
    df["y"] = np.random.default_rng(5).permutation(df["y"].to_numpy())   # target unrelated to features
    r = t.cv_baseline(_csv(tmp_path, "tr.csv", df), target="y", id_col="id")
    assert 0.40 < r["cv_scores"]["roc_auc"] < 0.60


def test_cv_regression_and_multiclass(tmp_path, binary):
    df = binary.copy()
    df["y_reg"] = 3 * df["x1"] + np.random.default_rng(2).normal(size=len(df))
    r = t.cv_baseline(_csv(tmp_path, "reg.csv", df.drop(columns="y")), target="y_reg", id_col="id")
    assert r["task"] == "regression" and r["cv_scores"]["rmse"] < 0.5 * r["no_skill_baseline"]["rmse"]
    df["y_mc"] = pd.cut(df["x1"], [-np.inf, -0.5, 0.5, np.inf], labels=["lo", "mid", "hi"]).astype(str)
    r = t.cv_baseline(_csv(tmp_path, "mc.csv", df.drop(columns=["y", "y_reg"])), target="y_mc", id_col="id")
    assert r["task"] == "multiclass" and r["cv_scores"]["accuracy"] > 0.8


# ── explain ───────────────────────────────────────────────────────────────
def test_explain_ranks_the_informative_feature_first(tmp_path, binary):
    r = t.explain_model(_csv(tmp_path, "tr.csv", binary), target="y", id_col="id")
    assert r["top_features"][0]["feature"] == "x1"
    noise = next(f for f in r["top_features"] if f["feature"] == "noise")
    assert noise["score_drop_mean"] < 0.02


# ── submission check ──────────────────────────────────────────────────────
@pytest.fixture
def sample(tmp_path):
    return _csv(tmp_path, "sample.csv", pd.DataFrame({"id": [1, 2, 3, 4], "is_fraud": 0.0}))


def test_good_submission_passes(tmp_path, sample):
    sub = _csv(tmp_path, "sub.csv", pd.DataFrame({"id": [1, 2, 3, 4], "is_fraud": [0.1, 0.9, 0.2, 0.05]}))
    assert t.check_submission(sub, sample)["ok"]


@pytest.mark.parametrize("bad, failing", [
    (pd.DataFrame({"id": [1, 2, 3, 3], "is_fraud": [0.1, 0.9, 0.2, 0.3]}), "no duplicated IDs"),
    (pd.DataFrame({"id": [1, 2, 3, 4], "is_fraud": [0.1, None, 0.2, 0.3]}), "'is_fraud' has no missing values"),
    (pd.DataFrame({"id": [1, 2, 3, 4], "is_fraud": [0.1, 1.7, 0.2, 0.3]}), "'is_fraud' stays within [0, 1]"),
    (pd.DataFrame({"id": [1, 2, 3, 4], "is_fraud": 0.5}), "'is_fraud' is not constant"),
    (pd.DataFrame({"id": [1, 2, 3, 4], "target": [0.1, 0.9, 0.2, 0.3]}), "same columns as sample"),
    (pd.DataFrame({"id": [1, 2, 3], "is_fraud": [0.1, 0.9, 0.2]}), "same number of rows"),
])
def test_bad_submissions_fail_the_right_check(tmp_path, sample, bad, failing):
    r = t.check_submission(_csv(tmp_path, "sub.csv", bad), sample)
    assert not r["ok"]
    assert any(c["check"] == failing and not c["ok"] for c in r["checks"])


# ── final fit + submission ────────────────────────────────────────────────
def test_fit_predict_writes_a_valid_submission(tmp_path, binary):
    tr = _csv(tmp_path, "tr.csv", binary.iloc[:3000])
    te = _csv(tmp_path, "te.csv", binary.iloc[3000:].drop(columns="y"))
    sample = _csv(tmp_path, "sample.csv", pd.DataFrame({"id": binary.id.iloc[3000:], "y": 0.0}))
    out = str(tmp_path / "submission.csv")
    r = t.fit_predict(tr, te, target="y", id_col="id", out_path=out)
    assert r["rows_predicted"] == 1000
    assert t.check_submission(out, sample, test_path=te)["ok"]
