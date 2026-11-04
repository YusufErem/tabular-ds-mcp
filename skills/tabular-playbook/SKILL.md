---
name: tabular-playbook
description: Inspect, profile, validate and baseline a tabular ML dataset (for example a Kaggle Playground competition) with the tabular-ds MCP tools, choosing random, time-ordered or group validation as the data requires, and write a short report in which every score sits next to a baseline. Use when the user gives a data folder or train/test CSV or Parquet files and asks what is in the data, whether the test set matches the training set, how well a first model does, or whether a submission file is valid.
---

# Tabular playbook

Use the `tabular-ds` MCP server. Work in this order and do not skip a step: each one can invalidate the next.

## 0. Inspect the folder
Call `inspect_competition(folder)` when you have a folder.
- Report `support`, the target, ID, task, submission format and `metric_hint`.
- Ask the user for the competition metric unless `settings` already holds one; pass it as `metric=` from then on.
- Every item in `limitations` goes into the report (joins, text, images, several targets).
- Use `suggested_calls` as the plan: they already carry `time_col` / `group_col` when the data needs them.

## 1. Profile the training table
Call `profile_dataset(path, target, id_col)`.
- Report rows, columns, target type and class balance.
- Every item in `warnings` goes into the report. Possible target leakage and ID-like columns must be resolved
  (ask the user) before any model is trained.

## 2. Compare train and test
Call `compare_train_test(train_path, test_path, target, id_col)`.
- `adversarial_auc` below 0.55: no detectable shift. 0.55–0.65: mild. Above 0.65: strong; name the top PSI
  features and say that cross-validation scores may overstate the test score. When `adversarial_auc_without_time`
  is present, judge the other features by it.
- Column or dtype mismatches and unseen categories are blocking issues: report them first.
- A `time_checks` verdict "test after train" means time-ordered validation (`time_col`) from now on. A
  `group_checks` verdict other than "mostly seen in train" means group validation (`group_col`).

## 3. Cross-validated baseline
Call `cv_baseline(train_path, target, id_col, metric, time_col | group_col)`.
- Always write `metric.model` **next to** `metric.no_skill` (for imbalanced binary targets also PR-AUC next to
  the positive rate). A score without its baseline is not reported.
- Report the fold spread; differences smaller than about two fold standard deviations are noise.
- If `random_split_comparison.verdict` is "random folds overstate the score", say by how much and use only the
  time-ordered / group score for decisions.

## 4. Explain
Call `explain_model(train_path, target, id_col, time_col | group_col)`.
- List the top features and the ones with no gain. Note that correlated features share importance.

## 5. Submission
Call `fit_predict(..., sample_submission_path, metric)` for a baseline file, then
`check_submission(submission_path, sample_submission_path, test_path)`.
- Do not tell the user a file is ready unless `ok` is true. Quote every failed check.

## Report format
Write `report.md` with six short sections (Problem, Data, Train vs test, Baseline, Important features,
Submission), each finding as one sentence with its number, and a final "Next steps" list. Never invent numbers:
every number in the report must come from a tool output in this session.
