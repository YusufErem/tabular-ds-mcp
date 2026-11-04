"""End-to-end: start the MCP server as a subprocess over stdio (as Claude Code does) and call it as a client."""
import json
import sys

import anyio
import numpy as np
import pandas as pd
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

TOOLS = {"inspect_competition", "profile_dataset", "compare_train_test", "cv_baseline", "explain_model",
         "fit_predict", "check_submission"}


def _payload(result):
    if getattr(result, "structured_content", None) or getattr(result, "structuredContent", None):
        return result.structured_content if hasattr(result, "structured_content") else result.structuredContent
    return json.loads(result.content[0].text)


def test_server_lists_and_runs_tools(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"id": range(600), "x": rng.normal(size=600)})
    df["y"] = (df["x"] > 1).astype(int)
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)

    async def run():
        params = StdioServerParameters(command=sys.executable, args=["-m", "tabular_ds_mcp.server"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = {tool.name for tool in (await session.list_tools()).tools}
                profile = _payload(await session.call_tool("profile_dataset",
                                                           {"path": str(path), "target": "y", "id_col": "id"}))
                cv = _payload(await session.call_tool("cv_baseline",
                                                      {"train_path": str(path), "target": "y", "id_col": "id",
                                                       "n_splits": 3}))
                return names, profile, cv

    names, profile, cv = anyio.run(run)
    assert names == TOOLS
    assert profile["rows"] == 600 and profile["target"]["task"] == "binary"
    assert cv["cv_scores"]["roc_auc"] > 0.95
