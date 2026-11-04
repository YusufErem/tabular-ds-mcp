"""tabular-ds-mcp: profile, validate and baseline tabular datasets, as plain Python or as an MCP server."""
from .core import (check_submission, compare_train_test, cv_baseline, explain_model, fit_predict,
                   inspect_competition, profile_dataset)

__all__ = ["inspect_competition", "profile_dataset", "compare_train_test", "cv_baseline", "explain_model",
           "fit_predict", "check_submission"]
__version__ = "0.2.0"
