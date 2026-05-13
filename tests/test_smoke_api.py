"""Smoke test — real API call to verify LLM can drive our tool protocol.

Reads LLM configuration from config/llm.toml. Tries [openrouter] first;
falls back to [mimo] if openrouter fails. Skips if the config file is missing.

This test calls a real paid API. It is explicitly opt-in via config file presence.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "llm.toml"


def _load_full_config() -> dict[str, Any]:
    """Load full config from config/llm.toml. Returns empty dict if missing."""
    if not CONFIG_PATH.is_file():
        return {}
    try:
        import tomli
    except ModuleNotFoundError:
        import tomllib as tomli  # type: ignore[no-redef]
    return tomli.loads(CONFIG_PATH.read_text())


def _make_client_from_config(full_config: dict[str, Any]) -> Any:
    """Try openrouter first, fall back to mimo. Returns (LLMClient, model_name) or None."""
    from research_agent.llm_client import LLMClient

    # Try openrouter
    or_cfg = full_config.get("openrouter", {})
    if or_cfg.get("api_key"):
        import os

        os.environ["OPENROUTER_API_KEY"] = or_cfg["api_key"]
        client = LLMClient(
            default_model=or_cfg["default_model"],
            proxy=or_cfg.get("proxy") or None,
        )
        try:
            resp = client.chat([{"role": "user", "content": "hi"}])
            if resp.content:
                return client, or_cfg["default_model"]
        except Exception:
            pass

    # Fallback to mimo
    mimo_cfg = full_config.get("mimo", {})
    if mimo_cfg.get("api_key"):
        # litellm needs openai/ prefix for custom OpenAI-compatible endpoints
        model = f"openai/{mimo_cfg['default_model']}"
        client = LLMClient(
            default_model=model,
            proxy=mimo_cfg.get("proxy") or None,
            api_base=mimo_cfg.get("api_base"),
            api_key=mimo_cfg["api_key"],
        )
        return client, model

    return None


FULL_CONFIG = _load_full_config()
CLIENT_INFO = _make_client_from_config(FULL_CONFIG) if FULL_CONFIG else None

skip_no_config = pytest.mark.skipif(
    CLIENT_INFO is None,
    reason="No working LLM provider in config/llm.toml — skipping real API test",
)


@skip_no_config
def test_llm_returns_valid_tool_call(tmp_path: Path) -> None:
    """Verify a real LLM can produce tool calls that our system parses correctly."""
    from research_agent.tool_registry import ToolRegistry

    assert CLIENT_INFO is not None
    client, _ = CLIENT_INFO

    registry = ToolRegistry()

    @registry.register(
        name="read_file",
        description="Read a file from the workspace.",
        parameters={"path": {"type": "string", "description": "File path"}},
    )
    def read_file(path: str) -> str:
        return f"Contents of {path}: sample data"

    response = client.chat(
        messages=[
            {"role": "system", "content": "You are a research agent. Use tools to complete tasks."},
            {"role": "user", "content": "Read the file data_catalog.json"},
        ],
        tools=registry.get_definitions(),
    )

    assert response.tool_calls, "LLM returned no tool calls"
    assert response.tool_calls[0].name == "read_file"
    assert "data_catalog" in response.tool_calls[0].arguments.get("path", "")


@skip_no_config
def test_harness_smoke_run(tmp_path: Path) -> None:
    """Run the harness with a trivial goal and small max_turns against a real LLM."""
    from research_agent.harness import ResearchHarness
    from research_agent.workspace import Workspace

    assert CLIENT_INFO is not None
    client, model = CLIENT_INFO

    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()

    (workspace.root / "data_catalog.json").write_text(
        '{"datasets": [{"name": "tcell_counts", "path": "data/tcell.csv"}]}',
        encoding="utf-8",
    )

    class DummyUserAgent:
        def respond(self, message: str, context: object | None = None) -> str:
            return "Acknowledged. Proceed with the plan."

    harness = ResearchHarness(
        model=model,
        workspace=workspace,
        user_agent=DummyUserAgent(),
        llm_client=client,
        max_turns=10,
    )

    result = harness.run("Read the data catalog and draft a plan for stage_01.")

    assert result.stop_reason in ("completed", "max_turns_reached")
    assert result.tool_call_count >= 1
    assert (workspace.root / "session.json").is_file()
