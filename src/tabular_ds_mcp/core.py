"""Plain-Python tools for profiling, validating and baselining tabular datasets.

Every function takes file paths (CSV or Parquet), returns a JSON-serialisable dict and never fits anything on
data it later evaluates on. The MCP server (server.py) exposes these functions to AI agents unchanged, and the
same functions can be imported in a notebook or a script.
"""
from __future__ import annotations

import itertools
import math
import re
import warnings as _warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, cohen_kappa_score, f1_score, log_loss,
                             matthews_corrcoef, mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold, StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

MAX_CAT_BINS = 250          # HistGradientBoosting handles at most 255 categories per feature
PSI_EPS = 1e-4
EPOCH = pd.Timestamp("1970-01-01")

DATE_RE = r"^\s*(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4})"
TIME_NAME_RE = re.compile(r"date|time|timestamp|year|month|week|period|step|(?:^|_)day(?:_|$)", re.I)
KEY_WORDS = r"customer|client|user|patient|store|shop|card|cc|account|device|session|household|group|item|product|merchant"
KEY_NAME_RE = re.compile(rf"(?:^|_)(?:{KEY_WORDS})?_?(?:id|key|no|num|number|code)$|^id_|^(?:{KEY_WORDS})s?$", re.I)
MEDIA_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".dcm", ".wav", ".mp3", ".flac", ".ogg",
             ".mp4", ".npy"}
MEDIA_RE = r"\.(?:jpe?g|png|tiff?|bmp|gif|dcm|wav|mp3|flac|ogg|mp4|npy)$"
TABLE_RE = re.compile(r"\.(?:csv|parquet|pq)(?:\.gz|\.zip)?$", re.I)

# metrics per task; 'map@k' is written map@3, map@5, ...
METRICS = {"binary": ["roc_auc", "pr_auc", "log_loss", "accuracy", "f1", "mcc"],
           "multiclass": ["log_loss", "accuracy", "macro_f1", "roc_auc_ovr", "qwk", "map@k"],
           "regression": ["rmse", "rmsle", "mae", "mape", "r2", "qwk"]}
STANDARD = {"binary": ["roc_auc", "pr_auc", "log_loss"], "multiclass": ["roc_auc_ovr", "log_loss", "accuracy"],
            "regression": ["rmse", "mae", "r2"]}
DEFAULT_METRIC = {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}
HIGHER_IS_BETTER = {"roc_auc", "pr_auc", "accuracy", "f1", "mcc", "macro_f1", "roc_auc_ovr", "qwk", "map@k", "r2"}
ALIASES = {"auc": "roc_auc", "rocauc": "roc_auc", "ap": "pr_auc", "average_precision": "pr_auc",
           "logloss": "log_loss", "cross_entropy": "log_loss", "f1_macro": "macro_f1", "kappa": "qwk",
           "quadratic_weighted_kappa": "qwk", "acc": "accuracy", "matthews": "mcc"}


# ═══════════════════════════ helpers ═══════════════════════════════════════
def _native(x: Any) -> Any:
    """Convert numpy / pandas scalars and containers to plain Python for JSON output."""
    if isinstance(x, dict):
        return {str(k): _native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_native(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return None if math.isnan(f) or math.isinf(f) else round(f, 6)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if x is pd.NA or x is pd.NaT:
        return None
    return x


def load_table(path: str, max_rows: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Read a CSV or Parquet file; optionally keep a random sample of `max_rows` rows."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if p.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return df


def _is_categorical(s: pd.Series) -> bool:
    return s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == bool or \
        pd.api.types.is_string_dtype(s.dtype)


def infer_task(y: pd.Series) -> str:
    """'binary', 'multiclass' or 'regression'."""
    n = y.nunique(dropna=True)
    if n == 2:
        return "binary"
    if _is_categorical(y) or (pd.api.types.is_integer_dtype(y) and n <= 20):
        return "multiclass"
    return "regression"


def _features(df: pd.DataFrame, target: str | None, id_col: str | None) -> list[str]:
    return [c for c in df.columns if c not in {target, id_col}]


# ── column detectors (row-wise rules, nothing is learned) ────────────────────
def _text_sample(s: pd.Series, n: int = 1000) -> pd.Series:
    return s.dropna().astype(str).head(n)


def _looks_like_date(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not _is_categorical(s) or s.dtype == bool:
        return False
    v = _text_sample(s)
    return len(v) > 0 and v.str.match(DATE_RE).mean() >= 0.95


def _looks_like_text(s: pd.Series) -> bool:
    if not _is_categorical(s) or _looks_like_date(s):
        return False
    v = _text_sample(s, 2000)
    return len(v) > 20 and v.str.len().mean() >= 30 and v.str.contains(" ").mean() >= 0.5 and \
        v.nunique() >= 0.3 * len(v)


def _looks_like_path(s: pd.Series) -> bool:
    if not _is_categorical(s):
        return False
    v = _text_sample(s)
    return len(v) > 0 and v.str.contains(MEDIA_RE, case=False).mean() >= 0.9


def _looks_like_key(s: pd.Series, name: str, n_rows: int) -> bool:
    """A repeated identifier (customer, store, card...) whose rows are not independent."""
    if pd.api.types.is_float_dtype(s) or not KEY_NAME_RE.search(str(name)):
        return False
    n = s.nunique(dropna=True)
    return 20 <= n <= 0.5 * n_rows


def _date_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if _looks_like_date(X[c])]


def _to_datetime(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        d = s
    else:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            d = pd.to_datetime(s, errors="coerce")
            if d.isna().sum() > s.isna().sum() + 0.05 * len(s):
                d = pd.to_datetime(s, errors="coerce", format="mixed")
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    return d


def _time_values(s: pd.Series) -> pd.Series:
    """A numeric time axis: days since 1970 for dates, the value itself for numeric columns."""
    if _looks_like_date(s):
        return (_to_datetime(s) - EPOCH).dt.total_seconds() / 86400
    return pd.to_numeric(s, errors="coerce")


def _show_time(v: float, is_date: bool) -> Any:
    if not is_date:
        return v
    return str((EPOCH + pd.to_timedelta(v, unit="D")).round("s")).replace(" 00:00:00", "")


def _clean(X: pd.DataFrame, date_cols: list[str] | tuple = ()) -> pd.DataFrame:
    """Row-wise only (nothing is learned): dates become numeric parts, categorical columns become strings."""
    X = X.copy()
    for c in date_cols:
        if c not in X.columns:
            continue
        d = _to_datetime(X[c])
        parts = {"days": (d - EPOCH).dt.total_seconds() / 86400, "year": d.dt.year, "month": d.dt.month,
                 "day": d.dt.day, "dayofweek": d.dt.day_of_week, "dayofyear": d.dt.day_of_year, "hour": d.dt.hour}
        loc = X.columns.get_loc(c)
        X = X.drop(columns=c)
        for j, (k, v) in enumerate(parts.items()):
            X.insert(loc + j, f"{c}__{k}", v.astype(float))
    for c in X.columns:
        if _is_categorical(X[c]):
            X[c] = X[c].astype(str)
    return X


def _pipeline(X: pd.DataFrame, task: str, seed: int) -> Pipeline:
    """Leakage-safe model: encoders are fitted inside the pipeline, i.e. only on each training fold."""
    cat = [c for c in X.columns if _is_categorical(X[c])]
    num = [c for c in X.columns if c not in cat]
    prep = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                encoded_missing_value=-1), cat),
         ("num", "passthrough", num)],
        verbose_feature_names_out=False)
    cat_mask = [X[c].nunique(dropna=True) <= MAX_CAT_BINS for c in cat] + [False] * len(num)
    params = dict(max_iter=300, learning_rate=0.08, early_stopping=True, validation_fraction=0.1,
                  n_iter_no_change=20, random_state=seed,
                  categorical_features=cat_mask if any(cat_mask) else None)
    model = HistGradientBoostingRegressor(**params) if task == "regression" else HistGradientBoostingClassifier(**params)
    return Pipeline([("prep", prep), ("model", model)])


def _encode_target(y: pd.Series, task: str) -> tuple[np.ndarray, list]:
    if task == "regression":
        return y.to_numpy(dtype=float), []
    values = y.dropna().unique().tolist()
    try:
        classes = sorted(values)                      # numeric classes keep their order (needed for qwk)
    except TypeError:
        classes = sorted(values, key=str)
    mapping = {c: i for i, c in enumerate(classes)}
    return y.map(mapping).to_numpy(dtype=int), classes


def _predict(m: Pipeline, X: pd.DataFrame, task: str, n_classes: int, log_target: bool = False) -> np.ndarray:
    """Binary: P(class 1). Multiclass: one column per class, also when a class was absent from the training fold."""
    if task == "regression":
        p = m.predict(X)
        return np.expm1(p) if log_target else p
    full = np.zeros((len(X), n_classes))
    full[:, np.asarray(m.classes_, dtype=int)] = m.predict_proba(X)
    return full[:, 1] if task == "binary" else full


# ── metrics ──────────────────────────────────────────────────────────────────
def _parse_metric(metric: str | None, task: str) -> tuple[str, int | None]:
    if metric is None:
        return DEFAULT_METRIC[task], None
    m = metric.strip().lower().replace("-", "_").replace(" ", "_")
    m = ALIASES.get(m, m)
    k = None
    hit = re.fullmatch(r"map_?@?_?(\d+)", m)
    if hit:
        m, k = "map@k", int(hit.group(1))
    if m == "roc_auc" and task == "multiclass":
        m = "roc_auc_ovr"
    if m == "f1" and task == "multiclass":
        m = "macro_f1"
    if m not in METRICS[task]:
        raise ValueError(f"Metric '{metric}' is not available for a {task} target. "
                         f"Available: {[x.replace('map@k', 'map@3') for x in METRICS[task]]}.")
    return m, k


def _label(name: str, k: int | None) -> str:
    return f"map@{k}" if name == "map@k" else name


def _metric_value(name: str, k: int | None, task: str, y: np.ndarray, pred: np.ndarray) -> float | None:
    """One metric; None when it is undefined on these rows (for example a single class)."""
    try:
        if task == "binary":
            if name == "roc_auc":
                return roc_auc_score(y, pred)
            if name == "pr_auc":
                return average_precision_score(y, pred)
            if name == "log_loss":
                return log_loss(y, np.clip(pred, 1e-15, 1 - 1e-15), labels=[0, 1])
            lab = (pred >= 0.5).astype(int)
            if name == "accuracy":
                return accuracy_score(y, lab)
            if name == "f1":
                return f1_score(y, lab, zero_division=0)
            return matthews_corrcoef(y, lab)
        if task == "multiclass":
            labels = list(range(pred.shape[1]))
            if name == "log_loss":
                p = np.clip(pred, 1e-15, 1)
                return log_loss(y, p / p.sum(1, keepdims=True), labels=labels)
            if name == "accuracy":
                return accuracy_score(y, pred.argmax(1))
            if name == "macro_f1":
                return f1_score(y, pred.argmax(1), average="macro", labels=labels, zero_division=0)
            if name == "roc_auc_ovr":
                return roc_auc_score(y, pred, multi_class="ovr", labels=labels)
            if name == "qwk":
                return cohen_kappa_score(y, pred.argmax(1), weights="quadratic")
            top = np.argsort(-pred, axis=1, kind="stable")[:, :k]
            return float(((top == y[:, None]) / np.arange(1, k + 1)).sum(1).mean())
        if name == "rmse":
            return math.sqrt(mean_squared_error(y, pred))
        if name == "rmsle":
            return math.sqrt(np.mean((np.log1p(np.clip(pred, 0, None)) - np.log1p(y)) ** 2))
        if name == "mae":
            return mean_absolute_error(y, pred)
        if name == "mape":
            return mean_absolute_percentage_error(y, pred)
        if name == "r2":
            return r2_score(y, pred)
        lo, hi = np.rint(y.min()), np.rint(y.max())
        return cohen_kappa_score(np.rint(y).astype(int), np.clip(np.rint(pred), lo, hi).astype(int),
                                 weights="quadratic")
    except ValueError:
        return None


def _scores(task: str, y: np.ndarray, pred: np.ndarray, primary: tuple[str, int | None] | None = None) -> dict:
    names = [(n, None) for n in STANDARD[task]]
    if primary and primary[0] not in STANDARD[task]:
        names.append(primary)
    return {_label(n, k): _metric_value(n, k, task, y, pred) for n, k in names}


# ── validation folds ─────────────────────────────────────────────────────────
def _folds(task: str, y: np.ndarray, n_splits: int, seed: int, times: np.ndarray | None = None,
           groups: np.ndarray | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Time-ordered expanding window, group K-fold or (stratified) K-fold."""
    idx = np.arange(len(y))
    if times is not None:
        u = np.unique(times)
        if len(u) < n_splits + 1:
            raise ValueError(f"The time column has only {len(u)} distinct values; use n_splits < {len(u)}.")
        blocks = np.array_split(u, n_splits + 1)      # equal blocks of distinct times: ties stay together
        return [(idx[times < b[0]], idx[(times >= b[0]) & (times <= b[-1])]) for b in blocks[1:]]
    dummy = np.zeros((len(y), 1))
    if groups is not None:
        splitter = (GroupKFold(n_splits, shuffle=True, random_state=seed) if task == "regression"
                    else StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed))
        return list(splitter.split(dummy, y, groups))
    splitter = (KFold(n_splits, shuffle=True, random_state=seed) if task == "regression"
                else StratifiedKFold(n_splits, shuffle=True, random_state=seed))
    return list(splitter.split(dummy, y))


def _cv_run(X: pd.DataFrame, y: np.ndarray, task: str, n_classes: int, folds: list, seed: int,
            log_target: bool = False) -> tuple[np.ndarray, np.ndarray, list]:
    """Out-of-fold predictions of the model and of the no-skill baseline; unscored rows stay NaN."""
    shape = (len(y), n_classes) if task == "multiclass" else (len(y),)
    oof, naive, used = np.full(shape, np.nan), np.full(shape, np.nan), []
    for i_tr, i_va in folds:
        if task != "regression" and len(np.unique(y[i_tr])) < 2:
            continue
        y_fit = np.log1p(y[i_tr]) if log_target else y[i_tr]
        m = _pipeline(X, task, seed).fit(X.iloc[i_tr], y_fit)
        oof[i_va] = _predict(m, X.iloc[i_va], task, n_classes, log_target)
        if task == "binary":
            naive[i_va] = y[i_tr].mean()
        elif task == "multiclass":
            naive[i_va] = np.bincount(y[i_tr], minlength=n_classes) / len(i_tr)
        else:
            naive[i_va] = np.expm1(y_fit.mean()) if log_target else y_fit.mean()
        used.append((i_tr, i_va))
    return oof, naive, used


def _scored(oof: np.ndarray) -> np.ndarray:
    return ~np.isnan(oof if oof.ndim == 1 else oof[:, 0])


# ═══════════════════════════ 0. inspect a competition folder ════════════════
def _peek(path: Path, n: int) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path).head(n)
    return pd.read_csv(path, nrows=n)


def _column(path: Path, col: str) -> pd.Series:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=[col])[col]
    return pd.read_csv(path, usecols=[col])[col]


def _count_rows(path: Path) -> int | None:
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
            return pq.ParquetFile(path).metadata.num_rows
        except Exception:
            return None
    if path.suffix.lower() != ".csv":
        return None
    n, last = 0, b"\n"
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            n, last = n + chunk.count(b"\n"), chunk[-1:]
    return max(n - 1 + (last != b"\n"), 0)


def _submission_format(ss: pd.DataFrame, id_col: str, task: str, classes: list) -> dict:
    """What the sample submission asks for: probability, value, label, top-k labels or class probabilities."""
    pred = [c for c in ss.columns if c != id_col]
    if len(pred) > 1:
        return {"kind": "class_probabilities", "columns": pred}
    s = ss[pred[0]]
    if pd.api.types.is_numeric_dtype(s) and s.dtype != bool:
        if task == "binary":
            kind = "label_or_probability" if pd.api.types.is_integer_dtype(s) else "probability"
        else:
            kind = "label" if task == "multiclass" else "value"
        return {"kind": kind, "columns": pred}
    tokens = s.dropna().astype(str).str.split()
    k = int(tokens.str.len().max()) if len(tokens) else 1
    flat = [t for row in tokens for t in row]
    known = {str(c) for c in classes}
    if k > 1 and flat and np.mean([t in known for t in flat]) >= 0.9:
        return {"kind": "top_k_labels", "columns": pred, "k": k}
    return {"kind": "label", "columns": pred}


def _metric_hint(task: str, fmt: dict, y: pd.Series) -> tuple[str, str | None]:
    """(hint for the user, metric to pass when the format leaves no doubt)."""
    kind = fmt["kind"]
    if kind == "top_k_labels":
        return f"map@{fmt['k']} (the sample asks for {fmt['k']} ranked labels per row)", f"map@{fmt['k']}"
    if kind == "class_probabilities":
        return "log_loss (one probability column per class)", "log_loss"
    if task == "binary":
        return {"probability": "roc_auc (most binary Playground episodes) or log_loss",
                "label_or_probability": "roc_auc if probabilities are accepted, otherwise accuracy / f1 / mcc",
                "label": "accuracy, f1 or mcc (the sample asks for labels)"}[kind], None
    if task == "multiclass":
        return "accuracy, macro_f1, or qwk if the classes are ordered", None
    hint = "rmse, rmsle or mae"
    yv = pd.to_numeric(y, errors="coerce").dropna()
    if len(yv) and yv.min() >= 0 and yv.skew() > 1:
        hint += f" (the target is non-negative and right-skewed, skew {yv.skew():.1f}: rmsle is common)"
    return hint, None


def _time_order(train: pd.Series, test: pd.Series) -> dict | None:
    """Where do the test rows sit on the training time axis?"""
    is_date = _looks_like_date(train)
    a, b = _time_values(train).dropna(), _time_values(test).dropna()
    if a.empty or b.empty:
        return None
    lo, hi = a.min(), a.max()
    after, before = float((b > hi).mean()), float((b < lo).mean())
    verdict = ("test after train" if after >= 0.9 else "test inside the train period" if after + before <= 0.1
               else "test partly outside the train period")
    return {"kind": "date" if is_date else "numeric", "train_range": [_show_time(lo, is_date), _show_time(hi, is_date)],
            "test_range": [_show_time(b.min(), is_date), _show_time(b.max(), is_date)],
            "test_share_after_train": after, "test_share_before_train": before, "verdict": verdict}


def _group_overlap(train: pd.Series, test: pd.Series) -> dict:
    share = float(test.isin(set(train.dropna().unique())).mean())
    verdict = ("mostly new in test" if share < 0.5 else "mostly seen in train" if share >= 0.9 else "partly new in test")
    return {"train_values": int(train.nunique()), "rows_per_value_train": len(train) / max(train.nunique(), 1),
            "test_rows_with_a_known_value": share, "verdict": verdict}


def inspect_competition(folder: str, peek_rows: int = 50_000) -> dict:
    """Start here: read a competition folder and say what kind of problem it is and how to call the other tools.

    Finds the train / test / sample-submission tables, the target and ID columns, the task, the submission
    format and a likely metric, time and group structure, extra tables that need a join, and non-tabular content.
    """
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")
    files = [p for p in itertools.islice(root.rglob("*"), 500_000) if p.is_file()]
    tables = sorted(p for p in files if TABLE_RE.search(p.name))
    media = sum(p.suffix.lower() in MEDIA_EXT for p in files)

    def stem(p: Path) -> str:
        return TABLE_RE.sub("", p.name).lower()

    def pick(cands: list[Path], word: str) -> Path | None:
        exact = [p for p in cands if stem(p) == word]
        return exact[0] if exact else (max(cands, key=lambda p: p.stat().st_size) if cands else None)

    sample = pick([p for p in tables if "sample" in stem(p) or "submission" in stem(p)], "sample_submission")
    rest = [p for p in tables if p != sample]
    train = pick([p for p in rest if "train" in stem(p)], "train")
    test = pick([p for p in rest if "test" in stem(p) and p != train], "test")
    notes, partial, settings = [], [], {}
    out: dict[str, Any] = {"folder": str(root), "files": {
        "train": str(train) if train else None, "test": str(test) if test else None,
        "sample_submission": str(sample) if sample else None, "n_tables": len(tables), "n_media_files": media}}
    if media:
        partial.append(f"{media} image / audio / array files: these tools only use the tabular columns.")
    if train is None:
        out.update(support="not supported", limitations=partial,
                   blockers=["No training table (a CSV/Parquet file named *train*) found."])
        return _native(out)

    tr = _peek(train, peek_rows)
    te = _peek(test, peek_rows) if test else None
    ss = _peek(sample, 2000) if sample else None
    out["files"]["train_rows"] = _count_rows(train)
    out["files"]["test_rows"] = _count_rows(test) if test else None
    if (out["files"]["train_rows"] or 0) > 200_000:
        notes.append(f"Train has {out['files']['train_rows']:,} rows; the tools sample 200,000 rows by default "
                     "(raise max_rows for the final run).")

    # ── target and ID ──
    id_col, targets, multi = None, [], False
    if ss is not None:
        id_col = ss.columns[0]
        pred_cols = list(ss.columns[1:])
        if pred_cols and all(c in tr.columns and (te is None or c not in te.columns) for c in pred_cols):
            targets, multi = pred_cols, len(pred_cols) > 1
    if not targets and te is not None:
        targets = [c for c in tr.columns if c not in te.columns and c != id_col]
    if id_col is None or id_col not in tr.columns:
        named = [c for c in tr.columns if str(c).lower() in {"id", "row_id", "index"}]
        id_col = named[0] if named else None
    out["id_col"] = id_col
    blockers = []
    if not targets:
        blockers.append("Could not find the target column (no test table or sample submission to compare with): "
                        "pass target=... yourself.")
    elif len(targets) > 1 and not multi:
        blockers.append(f"Several columns are only in train: {targets}. Say which one is the target.")
    if blockers:
        out.update(support="not supported", target_candidates=targets, blockers=blockers, notes=notes)
        return _native(out)
    target = targets[0]
    if multi:
        partial.append(f"{len(targets)} target columns {targets}: run cv_baseline once per target.")
    task = infer_task(tr[target])
    _, classes = _encode_target(tr[target], task)
    out.update(target=target, all_targets=targets, task=task)
    if task != "regression":
        out["classes"] = [str(c) for c in classes[:30]]

    # ── submission format and metric ──
    if multi:
        fmt = {"kind": "multi_target", "columns": targets}
    elif ss is not None:
        fmt = _submission_format(ss, id_col, task, classes)
    else:
        fmt = {"kind": "probability" if task == "binary" else "label" if task == "multiclass" else "value",
               "columns": [target]}
    out["submission_format"] = fmt
    hint, metric = (_metric_hint(task, fmt, tr[target]) if not multi
                    else ("one score per target column; see the Evaluation page", None))
    out["metric_hint"] = hint + ". Confirm it on the competition's Evaluation page and pass metric=..."
    if metric:
        settings["metric"] = metric

    # ── time structure ──
    feats = [c for c in _features(tr, target, id_col) if te is None or c in te.columns]
    date_cols = _date_columns(tr[feats])
    time_cands = date_cols + [c for c in feats if c not in date_cols and pd.api.types.is_numeric_dtype(tr[c])
                              and TIME_NAME_RE.search(str(c))]
    out["time_columns"] = []
    for c in time_cands:
        info = _time_order(_column(train, c), _column(test, c)) if test else None
        if info is None:
            if c in date_cols:
                out["time_columns"].append({"column": c, "kind": "date"})
            continue
        if c not in date_cols and info["verdict"] != "test after train":
            continue
        out["time_columns"].append({"column": c, **info})
        if info["verdict"] == "test after train" and "time_col" not in settings:
            settings["time_col"] = c
            notes.append(f"Test rows come after the training period in '{c}' (from {info['test_range'][0]}): "
                         f"validate with time-ordered folds, time_col='{c}'.")
        elif info["verdict"] == "test partly outside the train period":
            notes.append(f"'{c}': {100 * info['test_share_after_train']:.0f}% of test rows are later than any "
                         "training row; compare random and time-ordered folds.")

    # ── repeated keys (groups) ──
    out["group_columns"] = []
    for c in feats:
        if not _looks_like_key(tr[c], c, len(tr)):
            continue
        info = {"column": c, **(_group_overlap(_column(train, c), _column(test, c)) if test else {})}
        out["group_columns"].append(info)
        if info.get("verdict") == "mostly new in test" and "group_col" not in settings and "time_col" not in settings:
            settings["group_col"] = c
            notes.append(f"Most test rows have a '{c}' never seen in train: validate with group folds, "
                         f"group_col='{c}'.")

    # ── text and file-path columns ──
    for c in feats:
        if _looks_like_path(tr[c]):
            partial.append(f"'{c}' holds file paths (images / audio): the model sees it only as a category.")
        elif _looks_like_text(tr[c]):
            partial.append(f"'{c}' is free text: the baseline treats it as a high-cardinality category; "
                           "text features (TF-IDF, embeddings) would need to be added.")

    # ── extra tables ──
    out["extra_tables"] = []
    for p in tables:
        if p in {train, test, sample}:
            continue
        cols = list(_peek(p, 1000).columns)
        shared = [c for c in cols if c in tr.columns and c != target]
        info = {"file": str(p), "rows": _count_rows(p), "columns": cols[:30], "shared_with_train": shared}
        if set(_features(tr, None, id_col)) <= set(cols):
            info["role"] = "same columns as train (e.g. an original dataset): could be appended as extra rows"
        elif shared:
            key = shared[0]
            keys = _column(p, key)
            per_key = len(keys) / max(keys.nunique(), 1)
            info["role"] = (f"join on '{key}'" + (f" after aggregating ({per_key:.1f} rows per key)"
                                                  if per_key > 1.05 else " (one row per key)"))
            partial.append(f"{p.name} must be joined to train on '{key}' before using these tools.")
        else:
            info["role"] = "no shared column with train"
        out["extra_tables"].append(info)

    # ── verdict and the calls to make ──
    support = "partly supported" if partial else "supported with settings" if settings else "supported"
    common = {"target": target, "id_col": id_col}
    split = {k: settings[k] for k in ("time_col", "group_col") if k in settings}
    metric_arg = {"metric": settings["metric"]} if "metric" in settings else {}
    calls = [{"tool": "profile_dataset", "args": {"path": str(train), **common}}]
    if test:
        calls.append({"tool": "compare_train_test",
                      "args": {"train_path": str(train), "test_path": str(test), **common, **split}})
    calls.append({"tool": "cv_baseline", "args": {"train_path": str(train), **common, **split, **metric_arg}})
    calls.append({"tool": "explain_model", "args": {"train_path": str(train), **common, **split}})
    if test and sample:
        sub = "submission.csv"
        calls.append({"tool": "fit_predict",
                      "args": {"train_path": str(train), "test_path": str(test), **common, "out_path": sub,
                               "sample_submission_path": str(sample), **metric_arg}})
        calls.append({"tool": "check_submission",
                      "args": {"submission_path": sub, "sample_submission_path": str(sample),
                               "test_path": str(test), "id_col": id_col}})
    out.update(support=support, settings=settings, limitations=partial, notes=notes, suggested_calls=calls)
    return _native(out)


# ═══════════════════════════ 1. profile ════════════════════════════════════
def profile_dataset(path: str, target: str | None = None, id_col: str | None = None,
                    max_rows: int | None = None, max_columns_reported: int = 60) -> dict:
    """Shape, column types, missing values, cardinality, duplicates, target balance and warnings."""
    df = load_table(path, max_rows)
    if target and target not in df.columns:
        raise ValueError(f"Target column '{target}' not in file. Columns: {list(df.columns)[:30]}")
    feats = _features(df, target, id_col)
    warnings, cols = [], []
    for c in df.columns:
        s = df[c]
        n_unique = int(s.nunique(dropna=True))
        info = {"column": c, "dtype": str(s.dtype), "missing_pct": 100 * s.isna().mean(), "n_unique": n_unique}
        if pd.api.types.is_numeric_dtype(s) and not s.dtype == bool:
            d = s.describe()
            info.update(mean=d.get("mean"), std=d.get("std"), min=d.get("min"), max=d.get("max"))
        else:
            info["top_values"] = {str(k): int(v) for k, v in s.value_counts(dropna=True).head(5).items()}
        cols.append(info)
        if c in {target, id_col}:
            continue
        if info["missing_pct"] > 30:
            warnings.append(f"'{c}' is {info['missing_pct']:.1f}% missing.")
        if _looks_like_date(s):
            d = _to_datetime(s)
            info["kind"] = "date"
            warnings.append(f"'{c}' is a date ({d.min():%Y-%m-%d} to {d.max():%Y-%m-%d}); the model gets its numeric "
                            f"parts. If the test rows come later in time, pass time_col='{c}' to cv_baseline.")
            continue
        if _looks_like_path(s):
            info["kind"] = "file path"
            warnings.append(f"'{c}' holds file paths (images / audio): only the tabular columns are modelled.")
            continue
        if _looks_like_text(s):
            info["kind"] = "free text"
            warnings.append(f"'{c}' is free text: treated as a high-cardinality category; consider text features.")
            continue
        if n_unique <= 1:
            warnings.append(f"'{c}' is constant: drop it.")
        if n_unique == len(df) and not pd.api.types.is_float_dtype(s):
            warnings.append(f"'{c}' is unique for every row: probably an ID, exclude it from features.")
        if _is_categorical(s) and n_unique > 50:
            warnings.append(f"'{c}' is a high-cardinality categorical ({n_unique} values).")
        if _looks_like_key(s, c, len(df)):
            info["kind"] = "repeated key"
            warnings.append(f"'{c}' repeats ({len(df) / n_unique:.1f} rows per value): rows sharing it are not "
                            f"independent. If the test set has new values, pass group_col='{c}' to cv_baseline.")
    dup = int(df[feats + ([target] if target else [])].duplicated().sum()) if feats else 0
    if dup:
        warnings.append(f"{dup} duplicated rows (ignoring the ID column).")
    out = {"path": str(path), "rows": len(df), "columns": df.shape[1], "n_features": len(feats),
           "memory_mb": df.memory_usage(deep=True).sum() / 2 ** 20, "duplicate_rows": dup,
           "column_summary": cols[:max_columns_reported]}
    if target:
        y = df[target]
        task = infer_task(y)
        out["target"] = {"name": target, "task": task, "missing": int(y.isna().sum())}
        if task in {"binary", "multiclass"}:
            vc = y.value_counts(normalize=True)
            out["target"]["class_share"] = {str(k): v for k, v in vc.items()}
            if vc.min() < 0.05:
                warnings.append(f"Target is imbalanced (smallest class {100 * vc.min():.2f}%): report PR-AUC "
                                "next to ROC-AUC and use stratified folds.")
        else:
            out["target"].update(mean=y.mean(), std=y.std(), min=y.min(), max=y.max())
        # single-feature leakage screen: a lone numeric column that almost perfectly ranks the target
        if task == "binary":
            yb, _ = _encode_target(y, task)
            suspects = []
            for c in feats:
                s = df[c]
                if pd.api.types.is_numeric_dtype(s) and s.notna().all() and s.nunique() > 1:
                    auc = roc_auc_score(yb, s)
                    if max(auc, 1 - auc) > 0.98:
                        suspects.append(c)
            if suspects:
                warnings.append(f"Possible target leakage: {suspects} alone rank the target with AUC > 0.98.")
    out["warnings"] = warnings
    return _native(out)


# ═══════════════════════════ 2. train vs test ═══════════════════════════════
def _psi(train: pd.Series, test: pd.Series, bins: int = 10) -> float:
    if _is_categorical(train) or train.nunique() <= bins:
        top = set(train.value_counts().head(20).index)

        def cats(s):                       # missing values keep their own bin, rare values go to "__other__"
            s = s.astype(object)
            return s.where(s.isin(top) | s.isna(), "__other__").fillna("__missing__").astype(str)

        a, b = cats(train), cats(test)
        cats = sorted(set(a) | set(b))
        pa = a.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        pb = b.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    else:
        edges = np.unique(np.nanquantile(train.to_numpy(dtype=float), np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf

        def share(s):
            v = s.to_numpy(dtype=float)
            h = np.histogram(v[~np.isnan(v)], bins=edges)[0] / len(v)
            return np.append(h, np.isnan(v).mean())

        pa, pb = share(train), share(test)
    pa, pb = np.clip(pa, PSI_EPS, None), np.clip(pb, PSI_EPS, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def _adversarial(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str], seed: int) -> tuple[float, list[float]]:
    """Can a model tell train rows from test rows? Cross-validated AUC, 0.5 = indistinguishable."""
    n = min(len(tr), len(te))
    both = pd.concat([tr[cols].sample(n=n, random_state=seed), te[cols].sample(n=n, random_state=seed)],
                     ignore_index=True)
    X = _clean(both, _date_columns(both))
    y = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    aucs = []
    for i_tr, i_va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = _pipeline(X, "binary", seed).fit(X.iloc[i_tr], y[i_tr])
        aucs.append(roc_auc_score(y[i_va], m.predict_proba(X.iloc[i_va])[:, 1]))
    return float(np.mean(aucs)), aucs


def _shift_verdict(auc: float) -> str:
    return "no detectable shift" if auc < 0.55 else "mild shift" if auc < 0.65 else "strong shift"


def compare_train_test(train_path: str, test_path: str, target: str | None = None, id_col: str | None = None,
                       time_col: str | None = None, group_col: str | None = None, max_rows: int = 200_000,
                       seed: int = 0) -> dict:
    """Does the test set look like the training set? Column checks, PSI and KS per feature, adversarial AUC,
    and where the test rows sit in time and in repeated keys (customers, stores...)."""
    tr, te = load_table(train_path, max_rows, seed), load_table(test_path, max_rows, seed)
    f_tr, f_te = set(_features(tr, target, id_col)), set(_features(te, target, id_col))
    common = [c for c in _features(tr, target, id_col) if c in f_te]
    for c in (time_col, group_col):
        if c and c not in common:
            raise ValueError(f"Column '{c}' must be in both tables.")
    warnings = []
    only_train, only_test = sorted(f_tr - f_te), sorted(f_te - f_tr)
    if only_train or only_test:
        warnings.append(f"Column mismatch: only in train {only_train}, only in test {only_test}.")
    date_cols = _date_columns(tr[common])
    keys = [c for c in common if c == group_col or _looks_like_key(tr[c], c, len(tr))]
    per_feature = []
    for c in common:
        a, b = tr[c], te[c]
        if c in date_cols:
            a, b = _time_values(a), _time_values(b)
        row = {"feature": c, "psi": _psi(a, b)}
        if c in date_cols:
            row["kind"] = "date (compared as days)"
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b) and not _is_categorical(a):
            ks = ks_2samp(a.dropna(), b.dropna())
            row.update(ks_stat=ks.statistic, ks_p=ks.pvalue)
        else:
            unseen = set(b.dropna().unique()) - set(a.dropna().unique())
            row["unseen_test_categories"] = len(unseen)
            if unseen and c not in keys:
                warnings.append(f"'{c}': {len(unseen)} categories appear in test but never in train.")
        if str(tr[c].dtype) != str(te[c].dtype):
            warnings.append(f"'{c}' has dtype {tr[c].dtype} in train but {te[c].dtype} in test.")
        per_feature.append(row)
    per_feature.sort(key=lambda r: -r["psi"])

    # time: the given column, else every date column and numeric columns named like a time index
    time_checks, later = [], []
    cands = [time_col] if time_col else date_cols + [c for c in common if c not in date_cols and
                                                     pd.api.types.is_numeric_dtype(tr[c]) and TIME_NAME_RE.search(c)]
    for c in cands:
        info = _time_order(tr[c], te[c])
        if info is None or (c != time_col and c not in date_cols and info["verdict"] != "test after train"):
            continue
        time_checks.append({"column": c, **info})
        if info["verdict"] == "test after train":
            later.append(c)
            warnings.append(f"Test rows come after the training period in '{c}' "
                            f"({100 * info['test_share_after_train']:.0f}% are later than any training row): "
                            f"validate with cv_baseline(time_col='{c}'); random folds will look too optimistic.")
        elif info["verdict"] == "test partly outside the train period":
            warnings.append(f"'{c}': {100 * info['test_share_after_train']:.0f}% of test rows are later than any "
                            f"training row: compare random and time-ordered folds (time_col='{c}').")
    # groups: the given column, else columns that look like repeated keys
    group_checks = []
    for c in keys:
        info = {"column": c, **_group_overlap(tr[c], te[c])}
        group_checks.append(info)
        if info["verdict"] != "mostly seen in train":
            warnings.append(f"{100 * (1 - info['test_rows_with_a_known_value']):.0f}% of test rows have a '{c}' never "
                            f"seen in train: validate with cv_baseline(group_col='{c}') so a model cannot score by "
                            "memorising it.")

    adv, aucs = _adversarial(tr, te, common, seed)
    out = {"train_rows": len(tr), "test_rows": len(te), "n_common_features": len(common),
           "adversarial_auc": adv, "adversarial_auc_folds": aucs, "shift_verdict": _shift_verdict(adv)}
    rest = [c for c in common if c not in later]
    if later and rest:            # the time column separates train and test by design; judge the other features
        adv2, _ = _adversarial(tr, te, rest, seed)
        out.update(adversarial_auc_without_time=adv2, shift_verdict_without_time=_shift_verdict(adv2),
                   time_note=f"{later} separate train and test by design; the verdict without them describes "
                             "the other features.")
    big = [r["feature"] for r in per_feature if r["psi"] > 0.25 and r["feature"] not in later]
    if big:
        warnings.append(f"Large PSI (> 0.25) for {big}.")
    out.update(psi_guide="< 0.1 stable, 0.1–0.25 moderate, > 0.25 large", features_by_psi=per_feature[:30],
               time_checks=time_checks, group_checks=group_checks, warnings=warnings)
    return _native(out)


# ═══════════════════════════ 3. cross-validated baseline ════════════════════
def cv_baseline(train_path: str, target: str, id_col: str | None = None, task: str = "auto",
                metric: str | None = None, time_col: str | None = None, group_col: str | None = None,
                n_splits: int = 5, max_rows: int = 200_000, seed: int = 0, oof_path: str | None = None,
                compare_random: bool = True) -> dict:
    """K-fold gradient-boosting baseline next to a no-skill baseline. Preprocessing is fitted inside each fold.

    time_col: folds follow time (train on the past, score the next block). group_col: no value of this column is
    in both the training and the validation part of a fold. With either, the same model is also scored with
    random folds, and the gap shows how much random validation would overstate the score.
    """
    df = load_table(train_path, max_rows, seed)
    for c in (target, time_col, group_col):
        if c and c not in df.columns:
            raise ValueError(f"Column '{c}' not in file.")
    df = df[df[target].notna()]
    warnings, times = [], None
    if time_col:
        t = _time_values(df[time_col])
        if t.isna().all():
            raise ValueError(f"'{time_col}' could not be read as dates or numbers.")
        if t.isna().any():
            warnings.append(f"{int(t.isna().sum())} rows with a missing '{time_col}' were dropped.")
            df, t = df[t.notna()], t[t.notna()]
        order = np.argsort(t.to_numpy(), kind="stable")
        df, times = df.iloc[order], t.to_numpy()[order]
    df = df.reset_index(drop=True)
    task = infer_task(df[target]) if task == "auto" else task
    name, k = _parse_metric(metric, task)
    label, hib = _label(name, k), name in HIGHER_IS_BETTER
    log_target = name == "rmsle"
    if log_target and (df[target] < 0).any():
        raise ValueError("rmsle needs a non-negative target.")
    feats = [c for c in _features(df, target, id_col) if c != group_col]
    X = _clean(df[feats], _date_columns(df[feats]))
    y, classes = _encode_target(df[target], task)
    if time_col and group_col:
        warnings.append("Both time_col and group_col given: folds follow time; group_col is only left out of "
                        "the features.")
    groups = df[group_col].astype(str).to_numpy() if group_col and not time_col else None
    folds = _folds(task, y, n_splits, seed, times=times, groups=groups)
    oof, naive, used = _cv_run(X, y, task, len(classes), folds, seed, log_target)
    if len(used) < len(folds):
        warnings.append(f"{len(folds) - len(used)} early folds were skipped: their training part has one class only.")
    ok = _scored(oof)
    primary = (name, k)
    model = _scores(task, y[ok], oof[ok], primary)
    no_skill = {"description": ("predict the training-fold class share" if task != "regression"
                                else "predict the training-fold mean" + (" of log1p(target)" if log_target else ""))}
    no_skill.update(_scores(task, y[ok], naive[ok], primary))
    if task == "binary":
        no_skill.update(roc_auc=0.5, pr_auc=float(y[ok].mean()))
    is_date = bool(time_col) and _looks_like_date(df[time_col])
    fold_rows = []
    for j, (i_tr, i_va) in enumerate(used):
        row = {"fold": j + 1, "train_rows": len(i_tr), "valid_rows": len(i_va),
               **_scores(task, y[i_va], oof[i_va], primary)}
        if times is not None:
            row.update(valid_from=_show_time(times[i_va].min(), is_date),
                       valid_to=_show_time(times[i_va].max(), is_date))
        fold_rows.append(row)
    spread = {}
    for m in model:
        v = [f[m] for f in fold_rows if f[m] is not None]
        spread[f"{m}_fold_std"] = float(np.std(v)) if v else None
    scheme = ("time-ordered expanding window" if times is not None else "group k-fold" if groups is not None
              else "k-fold" if task == "regression" else "stratified k-fold")
    out = {"task": task, "rows": len(df), "n_features": X.shape[1],
           "validation": {"scheme": scheme, "n_splits": n_splits, "time_col": time_col, "group_col": group_col,
                          "rows_scored": int(ok.sum())},
           "model": "HistGradientBoosting (default-ish settings, early stopping)",
           "metric": {"name": label, "higher_is_better": hib, "model": model[label], "no_skill": no_skill[label],
                      "source": "given" if metric else "default"}}
    if not metric:
        warnings.append(f"No metric given: '{label}' is the main score. Pass the competition's metric "
                        f"(for a {task} target: {[x.replace('map@k', 'map@3') for x in METRICS[task]]}).")
    if log_target:
        out["metric"]["note"] = "The model is trained on log1p(target) and transformed back, which RMSLE rewards."
    if times is not None:
        out["validation"]["note"] = ("Each fold trains on all earlier rows and scores the next time block; the "
                                     "first block is only used for training.")
        f = [r[label] for r in fold_rows if r[label] is not None]
        if len(f) >= 3:
            early, last = np.array(f[:-1]), f[-1]
            worse = (early.mean() - last) if hib else (last - early.mean())
            if worse > max(2 * early.std(), 0.01 * abs(early.mean())):
                warnings.append(f"The latest fold scores {label} {last:.4f} vs {early.mean():.4f} on earlier folds, "
                                "although it trains on the most data: the relationship may be drifting over time. "
                                "Check recent periods and consider weighting recent rows.")
    if (times is not None or groups is not None) and compare_random:
        what = "time-ordered" if times is not None else "group"
        if times is not None:      # same rows and training size as each time fold, only the split is random
            rng = np.random.default_rng(seed)
            r_folds = []
            for i_tr, i_va in used:
                pool = rng.permutation(np.r_[i_tr, i_va])
                r_folds.append((np.sort(pool[len(i_va):]), np.sort(pool[:len(i_va)])))
            how = "each time-ordered fold repeated on the same rows with the same training size, split at random"
        else:
            r_folds = _folds(task, y, n_splits, seed)
            how = "random k-fold on the same rows"
        r_scores = []
        for i_tr, i_va in r_folds:
            o, _, u = _cv_run(X, y, task, len(classes), [(i_tr, i_va)], seed, log_target)
            r_scores.append(_metric_value(name, k, task, y[i_va], o[i_va]) if u else None)
        s_scores = [r[label] for r in fold_rows]
        sign = 1 if hib else -1
        if times is not None:      # paired by fold
            d = [sign * (r - s) for r, s in zip(r_scores, s_scores) if r is not None and s is not None]
            gap = float(np.mean(d)) if d else None
            se = float(np.std(d, ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0
        else:
            rs, ss = [v for v in r_scores if v is not None], [v for v in s_scores if v is not None]
            gap = sign * (np.mean(rs) - np.mean(ss)) if rs and ss else None
            se = math.sqrt(np.var(rs, ddof=1) / len(rs) + np.var(ss, ddof=1) / len(ss)) if len(rs) > 1 < len(ss) else 0.0
        if gap is not None:
            r_mean = float(np.mean([v for v in r_scores if v is not None]))
            s_mean = float(np.mean([v for v in s_scores if v is not None]))
            big = gap > max(2 * se, 0.01 * abs(s_mean), 1e-3)
            out["random_split_comparison"] = {
                "random_split": r_mean, what: s_mean, "random_looks_better_by": gap, "how": how,
                "scores": f"mean {label} over folds",
                "verdict": "random folds overstate the score" if big else "random and structured folds agree"}
            if big:
                warnings.append(f"Random splits give {label} {r_mean:.4f} but {what} folds give {s_mean:.4f}: random "
                                f"validation overstates the score by {abs(gap):.4f}. Select models with {what} folds.")
    out.update(cv_scores=model, fold_spread=spread, no_skill_baseline=no_skill, folds=fold_rows,
               classes=[str(c) for c in classes], oof_path=oof_path, warnings=warnings,
               note="Scores are out-of-fold; encoders and the model never see the rows they are scored on.")
    if oof_path:
        o = pd.DataFrame({id_col: df[id_col]} if id_col else {"row": np.arange(len(df))})
        o["y"] = y
        if oof.ndim == 1:
            o["oof_pred"] = oof
        else:
            for j, c in enumerate(classes):
                o[f"oof_{c}"] = oof[:, j]
        o.to_csv(oof_path, index=False)
    return _native(out)


# ═══════════════════════════ 4. explain ════════════════════════════════════
def explain_model(train_path: str, target: str, id_col: str | None = None, task: str = "auto",
                  time_col: str | None = None, group_col: str | None = None, top_k: int = 15,
                  max_rows: int = 100_000, n_repeats: int = 5, seed: int = 0) -> dict:
    """Permutation importance on a held-out split: how much the score drops when each feature is shuffled.
    The hold-out is the latest 20% in time with time_col, whole groups with group_col, else a random 20%."""
    df = load_table(train_path, max_rows, seed)
    df = df[df[target].notna()].reset_index(drop=True)
    task = infer_task(df[target]) if task == "auto" else task
    feats = [c for c in _features(df, target, id_col) if c != group_col]
    X = _clean(df[feats], _date_columns(df[feats]))
    y, classes = _encode_target(df[target], task)
    idx = np.arange(len(df))
    if time_col:
        t = _time_values(df[time_col]).to_numpy()
        cut = np.nanquantile(t, 0.8)
        i_tr, i_va, holdout = idx[t < cut], idx[t >= cut], f"latest 20% by '{time_col}'"
    elif group_col:
        i_tr, i_va = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed)
                          .split(X, y, df[group_col].astype(str)))
        holdout = f"20% of the '{group_col}' groups"
    else:
        i_tr, i_va = train_test_split(idx, test_size=0.2, random_state=seed,
                                      stratify=y if task != "regression" else None)
        holdout = "random 20%"
    m = _pipeline(X, task, seed).fit(X.iloc[i_tr], y[i_tr])
    scoring = {"binary": "roc_auc", "multiclass": "roc_auc_ovr", "regression": "neg_root_mean_squared_error"}[task]
    imp = permutation_importance(m, X.iloc[i_va], y[i_va], scoring=scoring, n_repeats=n_repeats, random_state=seed)
    order = np.argsort(-imp.importances_mean)
    rows = [{"feature": X.columns[i], "score_drop_mean": imp.importances_mean[i],
             "score_drop_std": imp.importances_std[i]} for i in order[:top_k]]
    pred = _predict(m, X.iloc[i_va], task, len(classes))
    useless = [X.columns[i] for i in order if imp.importances_mean[i] <= 0]
    return _native({"task": task, "scoring": scoring, "holdout": holdout,
                    "holdout_scores": _scores(task, y[i_va], pred),
                    "top_features": rows, "features_with_no_gain": useless[:30],
                    "note": "Importance = drop in the hold-out score when the feature is shuffled; correlated "
                            "features share importance. Date columns appear as their parts (date__year, ...)."})


# ═══════════════════════════ 5. submission check ════════════════════════════
def check_submission(submission_path: str, sample_submission_path: str, test_path: str | None = None,
                     id_col: str | None = None) -> dict:
    """Is the submission file safe to upload? Columns, row count, IDs, missing values and value formats."""
    sub, ref = load_table(submission_path), load_table(sample_submission_path)
    id_col = id_col or ref.columns[0]
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    def numeric(s):
        return pd.api.types.is_numeric_dtype(s) and s.dtype != bool

    add("same columns as sample", list(sub.columns) == list(ref.columns),
        f"submission {list(sub.columns)} vs sample {list(ref.columns)}")
    add("same number of rows", len(sub) == len(ref), f"{len(sub)} vs {len(ref)}")
    if id_col in sub.columns:
        add("no duplicated IDs", not sub[id_col].duplicated().any(), f"{int(sub[id_col].duplicated().sum())} duplicates")
        add("same ID set as sample", set(sub[id_col]) == set(ref[id_col]),
            f"{len(set(ref[id_col]) - set(sub[id_col]))} missing, {len(set(sub[id_col]) - set(ref[id_col]))} extra")
        same_order = len(sub) == len(ref) and (sub[id_col].to_numpy() == ref[id_col].to_numpy()).all()
        add("same ID order as sample (not always required)", same_order)
        if test_path:
            te = load_table(test_path)
            if id_col in te.columns:
                add("IDs match the test file", set(sub[id_col]) == set(te[id_col]))
    preds = [c for c in ref.columns if c != id_col and c in sub.columns]
    for c in preds:
        s, r = sub[c], ref[c]
        add(f"'{c}' has no missing values", s.notna().all(), f"{int(s.isna().sum())} missing")
        if numeric(r):
            if not numeric(s):
                add(f"'{c}' is numeric like the sample", False, f"dtype {s.dtype}")
                continue
            if len(preds) == 1:
                add(f"'{c}' is not constant", s.nunique() > 1, "a constant prediction carries no ranking")
            if r.between(0, 1).all():
                add(f"'{c}' stays within [0, 1]", s.between(0, 1).all(), f"min {s.min():.4g}, max {s.max():.4g}")
        else:                                     # labels, or space-separated ranked labels (MAP@k)
            txt = s.astype(str).str.strip()
            add(f"'{c}' has no empty labels", ((txt != "") & s.notna()).all())
            k = int(r.dropna().astype(str).str.split().str.len().max() or 1)
            if k > 1:
                tokens = txt.str.split()
                add(f"'{c}' has at most {k} labels per row", (tokens.str.len() <= k).all(),
                    f"max {int(tokens.str.len().max())}")
                rep = tokens.apply(lambda t: len(t) != len(set(t)))
                add(f"'{c}' repeats no label within a row", not rep.any(), f"{int(rep.sum())} rows")
    prob_cols = [c for c in preds if numeric(sub[c]) and numeric(ref[c]) and ref[c].between(0, 1).all()]
    if len(prob_cols) > 1:
        rs = sub[prob_cols].sum(axis=1)
        add("class probabilities sum to 1 per row (not always required)", np.allclose(rs, 1, atol=1e-3),
            f"row sums {rs.min():.4g} to {rs.max():.4g}")
    return _native({"ok": all(c["ok"] for c in checks if "not always required" not in c["check"]),
                    "id_column": id_col, "checks": checks})


# ═══════════════════════════ 6. final fit + submission ══════════════════════
def _class_columns(cols: list[str], classes: list, target: str) -> dict[str, int]:
    """Match sample-submission columns such as 'Status_C' or 'C' to the target classes."""
    mapping = {}
    for col in cols:
        for rule in (lambda s: col == s, lambda s: col == f"{target}_{s}", lambda s: col.endswith(f"_{s}")):
            hit = [j for j, c in enumerate(classes) if rule(str(c))]
            if len(hit) == 1:
                mapping[col] = hit[0]
                break
    missing = [c for c in cols if c not in mapping]
    if missing:
        raise ValueError(f"Could not match sample columns {missing} to the classes {[str(c) for c in classes]}.")
    return mapping


def fit_predict(train_path: str, test_path: str, target: str, id_col: str, out_path: str,
                sample_submission_path: str | None = None, task: str = "auto", metric: str | None = None,
                group_col: str | None = None, max_rows: int | None = None, seed: int = 0) -> dict:
    """Fit the leakage-safe pipeline on all training rows and write a submission in the sample's format:
    probability, value, label, space-separated top-k labels (MAP@k) or one probability column per class."""
    tr, te = load_table(train_path, max_rows, seed), load_table(test_path)
    tr = tr[tr[target].notna()].reset_index(drop=True)
    task = infer_task(tr[target]) if task == "auto" else task
    name, k = _parse_metric(metric, task)
    feats = [c for c in _features(tr, target, id_col) if c in te.columns and c != group_col]
    date_cols = _date_columns(tr[feats])
    y, classes = _encode_target(tr[target], task)
    log_target = name == "rmsle"
    X = _clean(tr[feats], date_cols)
    m = _pipeline(X, task, seed).fit(X, np.log1p(y) if log_target else y)
    pred = _predict(m, _clean(te[feats], date_cols), task, len(classes), log_target)
    ss = load_table(sample_submission_path) if sample_submission_path else None
    if ss is not None:
        fmt = _submission_format(ss, id_col, task, classes)
    else:
        fmt = {"kind": "top_k_labels" if name == "map@k" else "probability" if task == "binary"
               else "label" if task == "multiclass" else "value", "columns": [target], "k": k}
    if fmt["kind"] == "top_k_labels" and k:
        fmt["k"] = k
    out = pd.DataFrame({id_col: te[id_col].to_numpy()})
    col, kind = fmt["columns"][0], fmt["kind"]
    if kind == "class_probabilities":
        probs = np.c_[1 - pred, pred] if task == "binary" else pred
        for c, j in _class_columns(fmt["columns"], classes, target).items():
            out[c] = probs[:, j]
    elif kind == "top_k_labels":
        top = np.argsort(-pred, axis=1, kind="stable")[:, :fmt["k"]]
        out[col] = [" ".join(str(classes[j]) for j in row) for row in top]
    elif task == "binary" and (kind == "label" or (kind == "label_or_probability" and
                                                   name in {"accuracy", "f1", "mcc"})):
        out[col] = [classes[int(p >= 0.5)] for p in pred]
        kind = "label"
    elif task == "multiclass":
        out[col] = [classes[j] for j in pred.argmax(1)]
        kind = "label"
    else:
        out[col] = pred
        kind = "probability" if task == "binary" else "value"
    if ss is not None:
        out = out[[c for c in ss.columns if c in out.columns]]
    out.to_csv(out_path, index=False)
    return _native({"task": task, "rows_predicted": len(te), "features_used": len(feats), "written_as": kind,
                    "columns": list(out.columns), "trained_on_log1p": log_target, "out_path": out_path})
