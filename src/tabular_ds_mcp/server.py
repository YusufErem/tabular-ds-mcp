"""MCP server: exposes the tabular data-science tools in core.py to AI agents (Claude Code, Cursor, Claude Desktop).

Run over stdio:  tabular-ds-mcp   (or: python -m tabular_ds_mcp.server)
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from . import core

server = MCPServer(
    name="tabular-ds",
    instructions=(
        "Tools for profiling, validating and baselining tabular datasets stored as CSV or Parquet files. "
        "Start with inspect_competition on the data folder: it finds the target, ID, task, submission format, "
        "time or group structure and returns the calls to make with the right arguments. Typical order: "
        "inspect_competition -> profile_dataset -> compare_train_test -> cv_baseline -> explain_model -> "
        "fit_predict -> check_submission. Pass the competition's metric to cv_baseline, and time_col / group_col "
        "when the tools say the test rows are later in time or belong to new groups. Always report a model score "
        "next to the no-skill baseline, and treat warnings in the tool output as findings to report, not to hide."),
)


@server.tool()
def inspect_competition(folder: str) -> dict:
    """Start here. Read a competition or project folder and say what kind of problem it is: train / test /
    sample-submission files, target and ID columns, task, submission format and a likely metric, whether the
    test rows lie later in time or belong to new groups (customers, stores...), extra tables that need a join,
    image / text content, and the exact tool calls to make next.

    Args:
        folder: directory holding the competition files (for example /kaggle/input/<competition>).
    """
    return core.inspect_competition(folder)


@server.tool()
def profile_dataset(path: str, target: str | None = None, id_col: str | None = None,
                    max_rows: int | None = None) -> dict:
    """Profile a CSV/Parquet table: shape, column types, missing values, cardinality, duplicate rows,
    target balance and warnings (constant or ID-like columns, possible target leakage, date columns,
    repeated keys, free-text and file-path columns).

    Args:
        path: file path of the table.
        target: name of the target column, if the table has one.
        id_col: name of the ID column, which is excluded from the feature checks.
        max_rows: profile a random sample of this many rows (default: all rows).
    """
    return core.profile_dataset(path, target=target, id_col=id_col, max_rows=max_rows)


@server.tool()
def compare_train_test(train_path: str, test_path: str, target: str | None = None, id_col: str | None = None,
                       time_col: str | None = None, group_col: str | None = None, max_rows: int = 200_000) -> dict:
    """Check whether the test set looks like the training set: column and dtype mismatches, unseen
    categories, PSI and KS per feature, an adversarial-validation AUC (0.5 = indistinguishable), whether the
    test rows come after the training period, and how many test rows belong to keys never seen in train.

    Args:
        train_path: training table.
        test_path: test table (no target column needed).
        target: target column in the training table, excluded from the comparison.
        id_col: ID column, excluded from the comparison.
        time_col: time column to check (default: date columns are found automatically).
        group_col: key column to check, such as customer_id (default: found automatically).
        max_rows: sample size per table for the heavier checks.
    """
    return core.compare_train_test(train_path, test_path, target=target, id_col=id_col, time_col=time_col,
                                   group_col=group_col, max_rows=max_rows)


@server.tool()
def cv_baseline(train_path: str, target: str, id_col: str | None = None, task: str = "auto",
                metric: str | None = None, time_col: str | None = None, group_col: str | None = None,
                n_splits: int = 5, max_rows: int = 200_000, oof_path: str | None = None) -> dict:
    """Cross-validated gradient-boosting baseline reported next to a no-skill baseline. Encoders are
    fitted inside each training fold, so the scores are free of preprocessing leakage. With time_col the
    folds follow time; with group_col no group is split across folds. In both cases the same model is also
    scored with random folds, and the tool says whether random validation would overstate the score.

    Args:
        train_path: training table.
        target: target column.
        id_col: ID column, excluded from the features.
        task: 'auto', 'binary', 'multiclass' or 'regression'.
        metric: the competition metric. Binary: roc_auc, pr_auc, log_loss, accuracy, f1, mcc. Multiclass:
            log_loss, accuracy, macro_f1, roc_auc_ovr, qwk, map@3 (any k). Regression: rmse, rmsle, mae,
            mape, r2, qwk. Default: roc_auc / log_loss / rmse.
        time_col: date or time-index column; folds train on the past and score the next time block.
        group_col: key column (customer, store, patient...); a group never sits in both parts of a fold.
        n_splits: number of folds.
        max_rows: train on a random sample of this many rows.
        oof_path: optional CSV path to save out-of-fold predictions.
    """
    return core.cv_baseline(train_path, target, id_col=id_col, task=task, metric=metric, time_col=time_col,
                            group_col=group_col, n_splits=n_splits, max_rows=max_rows, oof_path=oof_path)


@server.tool()
def explain_model(train_path: str, target: str, id_col: str | None = None, task: str = "auto",
                  time_col: str | None = None, group_col: str | None = None, top_k: int = 15,
                  max_rows: int = 100_000) -> dict:
    """Which features matter? Permutation importance of a gradient-boosting model on a hold-out split
    (the latest 20% in time with time_col, 20% of the groups with group_col, else a random 20%).

    Args:
        train_path: training table.
        target: target column.
        id_col: ID column, excluded from the features.
        task: 'auto', 'binary', 'multiclass' or 'regression'.
        time_col: hold out the latest rows by this column.
        group_col: hold out whole groups of this column.
        top_k: number of features to return.
        max_rows: use a random sample of this many rows.
    """
    return core.explain_model(train_path, target, id_col=id_col, task=task, time_col=time_col,
                              group_col=group_col, top_k=top_k, max_rows=max_rows)


@server.tool()
def fit_predict(train_path: str, test_path: str, target: str, id_col: str, out_path: str,
                sample_submission_path: str | None = None, task: str = "auto", metric: str | None = None,
                group_col: str | None = None) -> dict:
    """Fit the baseline model on all training rows and write a submission file in the sample submission's
    format: one probability or value per row, a label, space-separated top-k labels (MAP@k) or one
    probability column per class. Run check_submission on the result before uploading.

    Args:
        train_path: training table.
        test_path: test table.
        target: target column.
        id_col: ID column, copied to the submission.
        out_path: where to write the submission CSV.
        sample_submission_path: the competition's sample submission (sets the output format).
        task: 'auto', 'binary', 'multiclass' or 'regression'.
        metric: the competition metric (rmsle trains on log1p(target); map@k writes k labels).
        group_col: key column to leave out of the features, as in cv_baseline.
    """
    return core.fit_predict(train_path, test_path, target, id_col, out_path,
                            sample_submission_path=sample_submission_path, task=task, metric=metric,
                            group_col=group_col)


@server.tool()
def check_submission(submission_path: str, sample_submission_path: str, test_path: str | None = None,
                     id_col: str | None = None) -> dict:
    """Check a submission file against the sample submission before uploading it: columns, row count,
    duplicated or missing IDs, missing values, constant predictions, the [0, 1] range for probabilities,
    per-row probability sums and the number of labels per row for top-k (MAP@k) submissions.

    Args:
        submission_path: the file to upload.
        sample_submission_path: the competition's sample submission.
        test_path: optional test table, to check that every test ID is predicted.
        id_col: ID column (default: first column of the sample submission).
    """
    return core.check_submission(submission_path, sample_submission_path, test_path=test_path, id_col=id_col)


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
