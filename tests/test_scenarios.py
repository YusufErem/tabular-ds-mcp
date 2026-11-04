"""Evaluation harness, part 2: time order, repeated groups, competition metrics, submission formats and
folder inspection. Each table is built so that the right answer is known in advance."""
import numpy as np
import pandas as pd
import pytest

import tabular_ds_mcp as t


def _csv(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def drifting(tmp_path):
    """Daily target that follows a random walk: neighbouring days are alike, the future is not predictable
    from the date. Random folds can interpolate between neighbours; time-ordered folds cannot."""
    rng = np.random.default_rng(0)
    days = pd.date_range("2019-01-01", periods=1200, freq="D")
    level = np.cumsum(rng.normal(0, 1, len(days)))
    day = np.repeat(np.arange(len(days)), 2)
    df = pd.DataFrame({"id": np.arange(len(day)), "date": days[day].strftime("%Y-%m-%d"),
                       "noise": rng.normal(size=len(day))})
    df["y"] = level[day] + rng.normal(0, 0.3, len(day))
    return df


@pytest.fixture
def customers():
    """Ten rows per customer; the target depends on a hidden customer effect, and 'signup_score' is a
    per-customer constant that identifies the customer but says nothing about new customers."""
    rng = np.random.default_rng(1)
    n_c, per = 400, 10
    effect, score = rng.normal(0, 2, n_c), rng.normal(size=n_c)
    cid = np.repeat(np.arange(n_c), per)
    df = pd.DataFrame({"id": np.arange(n_c * per), "customer_id": cid, "signup_score": score[cid],
                       "x": rng.normal(size=n_c * per)})
    df["y"] = (effect[cid] + 0.5 * df["x"] + rng.normal(0, 0.5, len(df)) > 1.5).astype(int)
    return df


@pytest.fixture
def classes5():
    rng = np.random.default_rng(2)
    n = 3000
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    score = x1 + 0.5 * x2 + rng.normal(0, 0.7, n)
    df = pd.DataFrame({"id": np.arange(n), "x1": x1, "x2": x2, "noise": rng.normal(size=n)})
    df["grade"] = pd.cut(score, [-np.inf, -1.2, -0.4, 0.4, 1.2, np.inf], labels=[1, 2, 3, 4, 5]).astype(int)
    df["label"] = df["grade"].map({1: "A", 2: "B", 3: "C", 4: "D", 5: "E"})
    return df


# ── time order ────────────────────────────────────────────────────────────
def test_time_folds_expose_optimistic_random_split(tmp_path, drifting):
    r = t.cv_baseline(_csv(tmp_path / "tr.csv", drifting), target="y", id_col="id", time_col="date", metric="rmse")
    assert r["validation"]["scheme"] == "time-ordered expanding window"
    cmp = r["random_split_comparison"]
    assert cmp["verdict"] == "random folds overstate the score"
    assert cmp["time-ordered"] > 2 * cmp["random_split"]                # rmse: lower is better
    assert any("overstate" in w for w in r["warnings"])
    folds = r["folds"]
    assert folds[0]["valid_from"] < folds[-1]["valid_from"]             # validation blocks move forward in time
    assert all(f["train_rows"] < g["train_rows"] for f, g in zip(folds, folds[1:]))


def test_time_folds_agree_with_random_when_rows_are_independent(tmp_path):
    rng = np.random.default_rng(6)
    n = 4000
    df = pd.DataFrame({"id": np.arange(n), "date": pd.date_range("2021-01-01", periods=n, freq="h")
                       .strftime("%Y-%m-%d %H:%M"), "x": rng.normal(size=n), "noise": rng.normal(size=n)})
    df["y"] = (df["x"] + rng.normal(0, 0.8, n) > 1).astype(int)      # rows independent of time
    r = t.cv_baseline(_csv(tmp_path / "tr.csv", df), target="y", id_col="id", time_col="date")
    assert r["random_split_comparison"]["verdict"] == "random and structured folds agree"


def test_compare_finds_future_test_period_and_new_customers(tmp_path, drifting, customers):
    tr, te = drifting.iloc[:1800], drifting.iloc[1800:].drop(columns="y")
    r = t.compare_train_test(_csv(tmp_path / "tr.csv", tr), _csv(tmp_path / "te.csv", te), target="y", id_col="id")
    check = next(c for c in r["time_checks"] if c["column"] == "date")
    assert check["verdict"] == "test after train" and check["test_share_after_train"] == 1.0
    assert any("time_col='date'" in w for w in r["warnings"])
    assert r["adversarial_auc_without_time"] < 0.6                     # the other feature does not shift
    c = customers
    tr, te = c[c.customer_id < 300], c[c.customer_id >= 300].drop(columns="y")
    r = t.compare_train_test(_csv(tmp_path / "ctr.csv", tr), _csv(tmp_path / "cte.csv", te), target="y", id_col="id")
    g = next(x for x in r["group_checks"] if x["column"] == "customer_id")
    assert g["verdict"] == "mostly new in test"
    assert any("group_col='customer_id'" in w for w in r["warnings"])


# ── groups ────────────────────────────────────────────────────────────────
def test_group_folds_expose_memorised_customers(tmp_path, customers):
    r = t.cv_baseline(_csv(tmp_path / "tr.csv", customers), target="y", id_col="id", group_col="customer_id")
    assert r["validation"]["scheme"] == "group k-fold"
    cmp = r["random_split_comparison"]
    assert cmp["verdict"] == "random folds overstate the score"
    assert cmp["random_split"] - cmp["group"] > 0.05


# ── profile hints ─────────────────────────────────────────────────────────
def test_profile_flags_date_key_text_and_path_columns(tmp_path, customers):
    df = customers.copy()
    rng = np.random.default_rng(3)
    df["date"] = pd.date_range("2022-01-01", periods=len(df), freq="h").strftime("%Y-%m-%d %H:%M:%S")
    df["review"] = [f"customer {i} wrote a fairly long review about item {rng.integers(1e6)}" for i in range(len(df))]
    df["photo"] = [f"images/{i}.jpg" for i in range(len(df))]
    w = " ".join(t.profile_dataset(_csv(tmp_path / "tr.csv", df), target="y", id_col="id")["warnings"])
    assert "'date' is a date" in w and "time_col='date'" in w
    assert "group_col='customer_id'" in w
    assert "'review' is free text" in w
    assert "'photo' holds file paths" in w
    assert "'date' is unique for every row" not in w                    # a timestamp is not an ID


# ── competition metrics ───────────────────────────────────────────────────
def test_map_at_3_and_qwk_for_multiclass(tmp_path, classes5):
    path = _csv(tmp_path / "tr.csv", classes5.drop(columns="grade"))
    r = t.cv_baseline(path, target="label", id_col="id", metric="MAP@3")
    assert r["metric"]["name"] == "map@3" and r["metric"]["higher_is_better"]
    assert r["metric"]["model"] > r["metric"]["no_skill"] + 0.2
    r = t.cv_baseline(_csv(tmp_path / "g.csv", classes5.drop(columns="label")), target="grade", id_col="id",
                      metric="quadratic weighted kappa")
    assert r["metric"]["name"] == "qwk" and r["metric"]["model"] > 0.6 > r["metric"]["no_skill"]


def test_rmsle_trains_on_log_target(tmp_path):
    rng = np.random.default_rng(4)
    x = rng.normal(size=3000)
    df = pd.DataFrame({"id": range(3000), "x": x, "price": np.exp(2 + 1.2 * x + rng.normal(0, 0.3, 3000))})
    r = t.cv_baseline(_csv(tmp_path / "tr.csv", df), target="price", id_col="id", metric="rmsle")
    assert "log1p" in r["metric"]["note"]
    assert r["metric"]["model"] < 0.5 * r["metric"]["no_skill"]


def test_unknown_or_mismatched_metric_is_refused(tmp_path, customers):
    path = _csv(tmp_path / "tr.csv", customers)
    with pytest.raises(ValueError, match="not available for a binary target"):
        t.cv_baseline(path, target="y", id_col="id", metric="map@3")
    r = t.cv_baseline(path, target="y", id_col="id", n_splits=3)
    assert any("No metric given" in w for w in r["warnings"])


# ── submission formats ────────────────────────────────────────────────────
@pytest.mark.parametrize("sample_value, metric, kind", [
    ({"label": "A B C"}, None, "top_k_labels"),
    ({"label": "A"}, None, "label"),
    ({"label_A": 0.2, "label_B": 0.2, "label_C": 0.2, "label_D": 0.2, "label_E": 0.2}, None, "class_probabilities"),
])
def test_fit_predict_follows_the_sample_format(tmp_path, classes5, sample_value, metric, kind):
    df = classes5.drop(columns="grade")
    tr = _csv(tmp_path / "tr.csv", df.iloc[:2500])
    te = _csv(tmp_path / "te.csv", df.iloc[2500:].drop(columns="label"))
    sample = _csv(tmp_path / "sample.csv", pd.DataFrame({"id": df.id.iloc[2500:], **sample_value}))
    out = str(tmp_path / "submission.csv")
    r = t.fit_predict(tr, te, target="label", id_col="id", out_path=out, sample_submission_path=sample, metric=metric)
    assert r["written_as"] == kind
    sub = pd.read_csv(out)
    assert list(sub.columns) == list(pd.read_csv(sample).columns)
    if kind == "top_k_labels":
        assert (sub["label"].str.split().str.len() == 3).all()
    if kind == "class_probabilities":
        assert np.allclose(sub.drop(columns="id").sum(axis=1), 1)
    check = t.check_submission(out, sample, test_path=te)
    assert check["ok"], [c for c in check["checks"] if not c["ok"]]


def test_check_submission_catches_bad_top_k_rows(tmp_path):
    sample = _csv(tmp_path / "sample.csv", pd.DataFrame({"id": [1, 2], "label": ["A B C", "A B C"]}))
    sub = _csv(tmp_path / "sub.csv", pd.DataFrame({"id": [1, 2], "label": ["A B C D", "A A B"]}))
    r = t.check_submission(sub, sample)
    failed = {c["check"] for c in r["checks"] if not c["ok"]}
    assert not r["ok"]
    assert "'label' has at most 3 labels per row" in failed and "'label' repeats no label within a row" in failed


# ── folder inspection ─────────────────────────────────────────────────────
def test_inspect_time_ordered_folder_with_extra_table(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    n = 3000
    df = pd.DataFrame({"id": np.arange(n),
                       "date": pd.date_range("2021-01-01", periods=n, freq="6h").strftime("%Y-%m-%d %H:%M"),
                       "store_id": rng.integers(0, 50, n), "x": rng.normal(size=n)})
    df["sold_out"] = (df["x"] + rng.normal(0, 1, n) > 1.5).astype(int)
    folder = tmp_path / "comp"
    _csv(folder / "train.csv", df.iloc[:2400])
    _csv(folder / "test.csv", df.iloc[2400:].drop(columns="sold_out"))
    _csv(folder / "sample_submission.csv", pd.DataFrame({"id": df.id.iloc[2400:], "sold_out": 0.5}))
    _csv(folder / "stores.csv", pd.DataFrame({"store_id": range(50), "size": rng.integers(1, 5, 50)}))
    r = t.inspect_competition(str(folder))
    assert (r["target"], r["id_col"], r["task"]) == ("sold_out", "id", "binary")
    assert r["submission_format"]["kind"] == "probability"
    assert r["settings"]["time_col"] == "date"
    assert r["support"] == "partly supported"
    assert any("stores.csv" in x and "store_id" in x for x in r["limitations"])
    cv = next(c for c in r["suggested_calls"] if c["tool"] == "cv_baseline")
    assert cv["args"]["time_col"] == "date"
    # every suggested call runs as written
    monkeypatch.chdir(tmp_path)
    for call in r["suggested_calls"]:
        res = getattr(t, call["tool"])(**call["args"])
        if call["tool"] == "check_submission":
            assert res["ok"]


def test_inspect_top_k_folder_and_missing_train(tmp_path, classes5):
    df = classes5.drop(columns="grade")
    folder = tmp_path / "comp"
    _csv(folder / "train.csv", df.iloc[:2500])
    _csv(folder / "test.csv", df.iloc[2500:].drop(columns="label"))
    _csv(folder / "sample_submission.csv", pd.DataFrame({"id": df.id.iloc[2500:], "label": "A B C"}))
    r = t.inspect_competition(str(folder))
    assert r["task"] == "multiclass" and r["submission_format"] == {"kind": "top_k_labels", "columns": ["label"], "k": 3}
    assert r["settings"] == {"metric": "map@3"} and r["support"] == "supported with settings"
    empty = tmp_path / "images_only"
    (empty / "train").mkdir(parents=True)
    (empty / "train" / "a.png").write_bytes(b"")
    r = t.inspect_competition(str(empty))
    assert r["support"] == "not supported" and r["files"]["n_media_files"] == 1


# ── output safety ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["notes.txt", "settings.json", "submission"])
def test_tools_only_write_csv_files(tmp_path, customers, name):
    tr = _csv(tmp_path / "tr.csv", customers.iloc[:3000])
    te = _csv(tmp_path / "te.csv", customers.iloc[3000:].drop(columns="y"))
    with pytest.raises(ValueError, match=".csv"):
        t.fit_predict(tr, te, target="y", id_col="id", out_path=str(tmp_path / name))
    assert not (tmp_path / name).exists()


def test_tools_refuse_to_write_through_a_symlink(tmp_path, customers):
    target = tmp_path / "important.csv"
    target.write_text("keep me")
    link = tmp_path / "oof.csv"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        t.cv_baseline(_csv(tmp_path / "tr.csv", customers), target="y", id_col="id", n_splits=3, oof_path=str(link))
    assert target.read_text() == "keep me"


# ── one-command demo ──────────────────────────────────────────────────────
def test_demo_runs_end_to_end(tmp_path, capsys):
    from tabular_ds_mcp import demo
    out = demo.run(str(tmp_path / "demo"), rows=4000)
    printed = capsys.readouterr().out
    assert out["check"]["ok"] and "[6/6]" in printed
    assert out["time"]["random_split_comparison"]["verdict"] == "random folds overstate the score"
    assert out["group"]["random_split_comparison"]["verdict"] == "random folds overstate the score"
