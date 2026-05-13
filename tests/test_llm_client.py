"""Tests for the litellm wrapper in research_agent.llm_client."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agent.llm_client import LLMClient, _PROXY_ENV_VARS
from research_agent.schemas import LLMResponse


def _build_response(
    *,
    content: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> SimpleNamespace:
    """Build a mock litellm response object with the expected attribute layout."""

    usage = None
    if prompt_tokens is not None or completion_tokens is not None:
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ],
        usage=usage,
    )


def test_basic_chat_returns_llm_response_with_content() -> None:
    """Chat should return normalized content when litellm returns plain text."""

    client = LLMClient(default_model="gpt-4o-mini")
    response = _build_response(content="hello")

    with patch("research_agent.llm_client.litellm.completion", return_value=response) as mock_completion:
        result = client.chat(messages=[{"role": "user", "content": "hi"}])

    assert isinstance(result, LLMResponse)
    assert result.content == "hello"
    assert result.tool_calls == []
    mock_completion.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
    )


def test_chat_with_tool_calls_converts_to_tool_call_requests() -> None:
    """Chat should parse litellm tool calls into ToolCallRequest models."""

    client = LLMClient(default_model="gpt-4o-mini")
    tool_calls = [
        SimpleNamespace(
            id="call_123",
            function=SimpleNamespace(
                name="search_paper",
                arguments='{"query": "transformers"}',
            ),
        )
    ]
    response = _build_response(content=None, tool_calls=tool_calls)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_paper",
                "description": "Search papers",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]

    with patch("research_agent.llm_client.litellm.completion", return_value=response):
        result = client.chat(messages=[{"role": "user", "content": "find papers"}], tools=tools)

    assert result.content is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_123"
    assert result.tool_calls[0].name == "search_paper"
    assert result.tool_calls[0].arguments == {"query": "transformers"}


def test_model_override_uses_explicit_model() -> None:
    """An explicit model argument should override the client's default model."""

    client = LLMClient(default_model="gpt-4o-mini")
    response = _build_response(content="ok")

    with patch("research_agent.llm_client.litellm.completion", return_value=response) as mock_completion:
        client.chat(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4",
        )

    mock_completion.assert_called_once_with(
        model="claude-sonnet-4",
        messages=[{"role": "user", "content": "hi"}],
    )


def test_tools_none_does_not_pass_tools_to_litellm() -> None:
    """tools=None should omit the tools argument from the litellm call."""

    client = LLMClient(default_model="gpt-4o-mini")
    response = _build_response(content="ok")

    with patch("research_agent.llm_client.litellm.completion", return_value=response) as mock_completion:
        client.chat(messages=[{"role": "user", "content": "hi"}], tools=None)

    _, kwargs = mock_completion.call_args
    assert "tools" not in kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_token_usage_is_captured_from_response_usage() -> None:
    """Prompt and completion token counts should be copied into LLMResponse."""

    client = LLMClient(default_model="gpt-4o-mini")
    response = _build_response(content="hello", prompt_tokens=10, completion_tokens=5)

    with patch("research_agent.llm_client.litellm.completion", return_value=response):
        result = client.chat(messages=[{"role": "user", "content": "hi"}])

    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5


# ---- Proxy isolation tests (2026-04-18 fix) ------------------------------
# Background: old code called os.environ.setdefault in __init__, which leaked
# proxies across clients. New code scopes proxy env per chat() via a context
# manager; construction is side-effect-free.


def _snapshot_proxy_env() -> dict[str, str | None]:
    return {var: os.environ.get(var) for var in _PROXY_ENV_VARS}


def _restore_proxy_env(snapshot: dict[str, str | None]) -> None:
    for var, prior in snapshot.items():
        if prior is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = prior


def test_construction_does_not_mutate_proxy_env() -> None:
    """Constructing LLMClient must never touch HTTP_PROXY/HTTPS_PROXY."""

    snapshot = _snapshot_proxy_env()
    try:
        # Clear all proxy vars so we can prove construction doesn't re-add them.
        for var in _PROXY_ENV_VARS:
            os.environ.pop(var, None)

        LLMClient(default_model="m", proxy="http://openrouter.example:8080")
        LLMClient(default_model="m", proxy=None)
        LLMClient(default_model="m", proxy="")

        for var in _PROXY_ENV_VARS:
            assert var not in os.environ, f"{var} leaked during construction"
    finally:
        _restore_proxy_env(snapshot)


def test_empty_proxy_string_normalized_to_none() -> None:
    """Empty-string proxy should be treated identically to None."""

    client = LLMClient(default_model="m", proxy="")
    assert client.proxy is None


def test_proxy_env_sets_and_restores_when_proxy_configured() -> None:
    """When proxy is set, _proxy_env() writes it and then restores prior value."""

    snapshot = _snapshot_proxy_env()
    try:
        os.environ["HTTP_PROXY"] = "http://shell-prior:1"
        os.environ["HTTPS_PROXY"] = "http://shell-prior:1"
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)

        client = LLMClient(default_model="m", proxy="http://client-proxy:9")
        with client._proxy_env():
            for var in _PROXY_ENV_VARS:
                assert os.environ[var] == "http://client-proxy:9"

        assert os.environ["HTTP_PROXY"] == "http://shell-prior:1"
        assert os.environ["HTTPS_PROXY"] == "http://shell-prior:1"
        assert "http_proxy" not in os.environ
        assert "https_proxy" not in os.environ
    finally:
        _restore_proxy_env(snapshot)


def test_proxy_env_pops_inherited_proxy_when_client_has_none() -> None:
    """proxy=None client must strip inherited proxy for the scope of the call.

    This is the MiMo case: shell has HTTPS_PROXY (for OpenRouter), but the
    MiMo client must call api.xiaomimimo.com WITHOUT that proxy. After the
    block, the shell's value must be restored so the next OpenRouter call
    still works.
    """

    snapshot = _snapshot_proxy_env()
    try:
        os.environ["HTTP_PROXY"] = "http://openrouter-proxy:8080"
        os.environ["HTTPS_PROXY"] = "http://openrouter-proxy:8080"
        os.environ["http_proxy"] = "http://openrouter-proxy:8080"
        os.environ["https_proxy"] = "http://openrouter-proxy:8080"

        mimo = LLMClient(default_model="mimo-v2-pro", proxy=None)
        with mimo._proxy_env():
            for var in _PROXY_ENV_VARS:
                assert var not in os.environ, f"{var} not popped for proxy=None client"

        for var in _PROXY_ENV_VARS:
            assert os.environ[var] == "http://openrouter-proxy:8080", f"{var} not restored"
    finally:
        _restore_proxy_env(snapshot)


def test_proxy_env_restores_on_exception() -> None:
    """finally block must restore env even if the wrapped block raises."""

    snapshot = _snapshot_proxy_env()
    try:
        os.environ["HTTP_PROXY"] = "http://outer:1"
        for var in ("HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ.pop(var, None)

        client = LLMClient(default_model="m", proxy="http://inner:2")
        try:
            with client._proxy_env():
                assert os.environ["HTTP_PROXY"] == "http://inner:2"
                raise RuntimeError("boom")
        except RuntimeError:
            pass

        assert os.environ["HTTP_PROXY"] == "http://outer:1"
        for var in ("HTTPS_PROXY", "http_proxy", "https_proxy"):
            assert var not in os.environ
    finally:
        _restore_proxy_env(snapshot)


def test_chat_wraps_completion_in_proxy_env() -> None:
    """chat() must invoke litellm.completion inside the proxy context.

    For a proxy=None client, HTTP_PROXY must be absent at the moment
    completion() is called, even if the shell had it set.
    """

    snapshot = _snapshot_proxy_env()
    try:
        os.environ["HTTPS_PROXY"] = "http://shell-proxy:1"

        observed: dict[str, str | None] = {}

        def fake_completion(**kwargs):
            observed["https"] = os.environ.get("HTTPS_PROXY")
            observed["http"] = os.environ.get("HTTP_PROXY")
            return _build_response(content="ok")

        mimo = LLMClient(default_model="m", proxy=None)
        with patch("research_agent.llm_client.litellm.completion", side_effect=fake_completion):
            mimo.chat(messages=[{"role": "user", "content": "hi"}])

        assert observed["https"] is None, "shell HTTPS_PROXY leaked into MiMo call"
        assert observed["http"] is None, "shell HTTP_PROXY leaked into MiMo call"
        # Shell env restored after the call.
        assert os.environ["HTTPS_PROXY"] == "http://shell-proxy:1"
    finally:
        _restore_proxy_env(snapshot)
