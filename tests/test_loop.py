"""Tests for the core agentic loop."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agent.llm_client import LLMClient
from research_agent.loop import run_agent_loop
from research_agent.schemas import LLMResponse
from research_agent.schemas import ToolCallRequest
from research_agent.tool_registry import ToolRegistry


def test_run_agent_loop_completes_without_tool_calls() -> None:
    """Loop should stop as completed when the LLM returns plain content."""

    client = MagicMock(spec=LLMClient)
    client.chat.return_value = LLMResponse(
        content="Final answer",
        tool_calls=[],
        prompt_tokens=10,
        completion_tokens=5,
    )
    messages: list[dict[str, object]] = [{"role": "user", "content": "Hello"}]
    registry = ToolRegistry()

    result = run_agent_loop(
        llm_client=client,
        messages=messages,
        tools=registry,
        max_turns=3,
    )

    assert result.stop_reason == "completed"
    assert result.final_response == "Final answer"
    assert result.turn_count == 0
    assert result.tool_call_count == 0
    assert messages == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Final answer"},
    ]


def test_run_agent_loop_rejects_plain_text_completion_when_terminal_tool_required() -> None:
    """Main-agent mode should reject plain-text completion without a terminal tool."""

    client = MagicMock(spec=LLMClient)
    client.chat.return_value = LLMResponse(
        content="Final answer",
        tool_calls=[],
    )
    messages: list[dict[str, object]] = [{"role": "user", "content": "Hello"}]
    registry = ToolRegistry()

    result = run_agent_loop(
        llm_client=client,
        messages=messages,
        tools=registry,
        max_turns=3,
        require_terminal_tool=True,
    )

    assert result.stop_reason == "error"
    assert "terminal tool" in (result.error_message or "")
    assert messages == [{"role": "user", "content": "Hello"}]


def test_run_agent_loop_executes_tool_then_completes() -> None:
    """Loop should execute tool calls, append results, and continue to completion."""

    client = MagicMock(spec=LLMClient)
    client.chat.side_effect = [
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="1", name="my_tool", arguments={"x": "1"})],
            prompt_tokens=10,
            completion_tokens=5,
        ),
        LLMResponse(
            content="Done",
            tool_calls=[],
            prompt_tokens=15,
            completion_tokens=8,
        ),
    ]

    registry = ToolRegistry()

    @registry.register(
        name="my_tool",
        description="test",
        parameters={"x": {"type": "string", "description": "Input value"}},
    )
    def my_tool(x: str) -> str:
        return f"result: {x}"

    messages: list[dict[str, object]] = [{"role": "user", "content": "Run"}]

    result = run_agent_loop(
        llm_client=client,
        messages=messages,
        tools=registry,
        max_turns=3,
    )

    assert result.stop_reason == "completed"
    assert result.final_response == "Done"
    assert result.turn_count == 1
    assert result.tool_call_count == 1
    assert messages == [
        {"role": "user", "content": "Run"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {
                        "name": "my_tool",
                        "arguments": json.dumps({"x": "1"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "1", "content": "result: 1"},
        {"role": "assistant", "content": "Done"},
    ]


def test_run_agent_loop_stops_when_max_turns_reached() -> None:
    """Loop should stop once the configured tool-turn limit is reached."""

    client = MagicMock(spec=LLMClient)
    client.chat.side_effect = [
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="1", name="my_tool", arguments={"x": "1"})],
        ),
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="2", name="my_tool", arguments={"x": "2"})],
        ),
    ]

    registry = ToolRegistry()

    @registry.register(
        name="my_tool",
        description="test",
        parameters={"x": {"type": "string", "description": "Input value"}},
    )
    def my_tool(x: str) -> str:
        return f"result: {x}"

    messages: list[dict[str, object]] = [{"role": "user", "content": "Run"}]

    result = run_agent_loop(
        llm_client=client,
        messages=messages,
        tools=registry,
        max_turns=1,
    )

    assert result.stop_reason == "max_turns_reached"
    assert result.turn_count == 1
    assert result.tool_call_count == 1
    assert len(messages) == 3


def test_run_agent_loop_executes_multiple_tool_calls_in_one_turn() -> None:
    """Loop should execute and append results for all tool calls in a single turn."""

    client = MagicMock(spec=LLMClient)
    client.chat.side_effect = [
        LLMResponse(
            content="Need two tools",
            tool_calls=[
                ToolCallRequest(id="1", name="my_tool", arguments={"x": "1"}),
                ToolCallRequest(id="2", name="my_tool", arguments={"x": "2"}),
            ],
        ),
        LLMResponse(content="Done", tool_calls=[]),
    ]

    registry = ToolRegistry()

    @registry.register(
        name="my_tool",
        description="test",
        parameters={"x": {"type": "string", "description": "Input value"}},
    )
    def my_tool(x: str) -> str:
        return f"result: {x}"

    messages: list[dict[str, object]] = [{"role": "user", "content": "Run"}]

    result = run_agent_loop(
        llm_client=client,
        messages=messages,
        tools=registry,
        max_turns=3,
    )

    assert result.stop_reason == "completed"
    assert result.tool_call_count == 2
    assert result.turn_count == 1
    assert messages[2] == {"role": "tool", "tool_call_id": "1", "content": "result: 1"}
    assert messages[3] == {"role": "tool", "tool_call_id": "2", "content": "result: 2"}


def test_run_agent_loop_stops_when_terminal_tool_is_called() -> None:
    """A terminal tool should complete the loop without waiting for another LLM turn."""

    client = MagicMock(spec=LLMClient)
    client.chat.return_value = LLMResponse(
        content="Finishing up.",
        tool_calls=[ToolCallRequest(id="1", name="finish_run", arguments={"summary": "Done"})],
    )

    registry = ToolRegistry()

    @registry.register(
        name="finish_run",
        description="Complete the workflow.",
        parameters={"summary": {"type": "string", "description": "Final summary"}},
        terminal=True,
    )
    def finish_run(summary: str) -> str:
        return summary

    messages: list[dict[str, object]] = [{"role": "user", "content": "Run"}]

    result = run_agent_loop(
        llm_client=client,
        messages=messages,
        tools=registry,
        max_turns=3,
        require_terminal_tool=True,
    )

    assert result.stop_reason == "completed"
    assert result.final_response == "Done"
    assert result.turn_count == 1
    assert result.tool_call_count == 1
    assert messages[-1] == {"role": "tool", "tool_call_id": "1", "content": "Done"}


def test_run_agent_loop_propagates_unexpected_tool_errors() -> None:
    """Programming bugs in tool implementations should propagate as exceptions, not be caught."""

    client = MagicMock(spec=LLMClient)
    client.chat.return_value = LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id="1", name="explode", arguments={"x": "1"})],
        prompt_tokens=7,
        completion_tokens=3,
    )

    registry = ToolRegistry()

    @registry.register(
        name="explode",
        description="test",
        parameters={"x": {"type": "string", "description": "Input value"}},
    )
    def explode(x: str) -> str:
        raise RuntimeError(f"boom: {x}")

    messages: list[dict[str, object]] = [{"role": "user", "content": "Run"}]

    with pytest.raises(RuntimeError, match="boom: 1"):
        run_agent_loop(
            llm_client=client,
            messages=messages,
            tools=registry,
            max_turns=3,
        )


def test_run_agent_loop_returns_error_when_trap_errors_enabled() -> None:
    """trap_errors should convert unexpected exceptions into an error stop reason."""

    client = MagicMock(spec=LLMClient)
    client.chat.return_value = LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id="1", name="explode", arguments={"x": "1"})],
    )

    registry = ToolRegistry()

    @registry.register(
        name="explode",
        description="Raise a bug-like exception.",
        parameters={"x": {"type": "string", "description": "Input value"}},
    )
    def explode(x: str) -> str:
        raise RuntimeError(f"boom: {x}")

    messages: list[dict[str, object]] = [{"role": "user", "content": "Run"}]

    result = run_agent_loop(
        llm_client=client,
        messages=messages,
        tools=registry,
        max_turns=3,
        trap_errors=True,
    )

    assert result.stop_reason == "error"
    assert result.error_message == "RuntimeError: boom: 1"
    assert result.final_response == "RuntimeError: boom: 1"


def test_run_agent_loop_truncates_tool_results_when_truncate_fn_provided() -> None:
    """truncate_fn is applied to tool results before they are appended to messages."""

    client = MagicMock(spec=LLMClient)
    client.chat.side_effect = [
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="1", name="my_tool", arguments={"x": "1"})],
        ),
        LLMResponse(content="Done", tool_calls=[]),
    ]

    registry = ToolRegistry()

    @registry.register(
        name="my_tool",
        description="test",
        parameters={"x": {"type": "string", "description": "Input value"}},
    )
    def my_tool(x: str) -> str:
        return "A" * 20_000

    messages: list[dict[str, object]] = [{"role": "user", "content": "Run"}]

    def truncate(content: str) -> str:
        if len(content) > 100:
            return content[:100] + "...[truncated]"
        return content

    result = run_agent_loop(
        llm_client=client,
        messages=messages,
        tools=registry,
        max_turns=3,
        truncate_fn=truncate,
    )

    assert result.stop_reason == "completed"
    tool_msg = messages[2]
    assert tool_msg["role"] == "tool"
    assert len(tool_msg["content"]) == 100 + len("...[truncated]")


def test_run_agent_loop_appends_state_overlay_transiently_each_turn() -> None:
    """state_overlay_fn output is appended to the request only, not to messages.

    Invariant: messages must never grow with overlay payloads — those are
    transient per-turn injections that go away after the LLM call. This
    keeps the persistent system prefix byte-stable for prompt caching.
    """

    call_count = 0
    seen_messages: list[list[dict[str, object]]] = []

    def overlay() -> str:
        nonlocal call_count
        call_count += 1
        return f"State v{call_count}"

    client = MagicMock(spec=LLMClient)

    def fake_chat(messages, *args, **kwargs):
        seen_messages.append([dict(m) for m in messages])
        if call_count == 1:
            return LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="1", name="my_tool", arguments={"x": "1"})],
            )
        return LLMResponse(content="Done", tool_calls=[])

    client.chat.side_effect = fake_chat

    registry = ToolRegistry()

    @registry.register(
        name="my_tool",
        description="test",
        parameters={"x": {"type": "string", "description": "Input value"}},
    )
    def my_tool(x: str) -> str:
        return f"result: {x}"

    messages: list[dict[str, object]] = [
        {"role": "system", "content": "System base"},
        {"role": "user", "content": "Run"},
    ]

    run_agent_loop(
        llm_client=client,
        messages=messages,
        tools=registry,
        max_turns=3,
        state_overlay_fn=overlay,
    )

    # Overlay must be invoked once per LLM call.
    assert call_count == 2

    # Persisted messages MUST NOT contain any overlay payload.
    assert messages[0] == {"role": "system", "content": "System base"}
    persisted_contents = [m.get("content") for m in messages]
    assert not any(
        isinstance(c, str) and c.startswith("State v") for c in persisted_contents
    )

    # The first request snapshot must include the overlay as a trailing user msg.
    first_request = seen_messages[0]
    assert first_request[-1] == {"role": "user", "content": "State v1"}
    # And the second request must include the *new* overlay, not v1.
    second_request = seen_messages[1]
    assert second_request[-1] == {"role": "user", "content": "State v2"}


def test_run_agent_loop_accumulates_token_usage_across_turns() -> None:
    """Prompt and completion token usage should be summed across loop iterations."""

    client = MagicMock(spec=LLMClient)
    client.chat.side_effect = [
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="1", name="my_tool", arguments={"x": "1"})],
            prompt_tokens=10,
            completion_tokens=5,
        ),
        LLMResponse(
            content="Done",
            tool_calls=[],
            prompt_tokens=15,
            completion_tokens=8,
        ),
    ]

    registry = ToolRegistry()

    @registry.register(
        name="my_tool",
        description="test",
        parameters={"x": {"type": "string", "description": "Input value"}},
    )
    def my_tool(x: str) -> str:
        return f"result: {x}"

    messages: list[dict[str, object]] = [{"role": "user", "content": "Run"}]

    result = run_agent_loop(
        llm_client=client,
        messages=messages,
        tools=registry,
        max_turns=3,
    )

    assert result.prompt_tokens == 25
    assert result.completion_tokens == 13


def test_run_agent_loop_invokes_compact_fn_when_prompt_tokens_exceed_threshold() -> None:
    """compact_fn must fire after a turn whose prompt_tokens exceed the threshold."""

    client = MagicMock(spec=LLMClient)
    client.chat.side_effect = [
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="1", name="my_tool", arguments={"x": "1"})],
            prompt_tokens=5_000,  # below threshold
        ),
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="2", name="my_tool", arguments={"x": "2"})],
            prompt_tokens=20_000,  # above threshold — should trigger compaction
        ),
        LLMResponse(content="Done", tool_calls=[], prompt_tokens=8_000),
    ]

    registry = ToolRegistry()

    @registry.register(
        name="my_tool",
        description="test",
        parameters={"x": {"type": "string", "description": "Input value"}},
    )
    def my_tool(x: str) -> str:
        return f"result: {x}"

    compact_fn = MagicMock()
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "Run"},
    ]

    run_agent_loop(
        llm_client=client,
        messages=messages,
        tools=registry,
        max_turns=5,
        compact_fn=compact_fn,
        compaction_threshold_tokens=10_000,
    )

    # Compact only fires after the 20k-token turn.
    assert compact_fn.call_count == 1


def test_run_agent_loop_skips_compact_fn_when_threshold_zero() -> None:
    """Threshold=0 (the default) must disable compaction entirely."""

    client = MagicMock(spec=LLMClient)
    client.chat.side_effect = [
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="1", name="my_tool", arguments={"x": "1"})],
            prompt_tokens=99_999,  # huge, but threshold is 0 → no compaction
        ),
        LLMResponse(content="Done", tool_calls=[]),
    ]

    registry = ToolRegistry()

    @registry.register(
        name="my_tool",
        description="test",
        parameters={"x": {"type": "string", "description": "Input value"}},
    )
    def my_tool(x: str) -> str:
        return f"result: {x}"

    compact_fn = MagicMock()
    messages: list[dict[str, object]] = [{"role": "user", "content": "Run"}]

    run_agent_loop(
        llm_client=client,
        messages=messages,
        tools=registry,
        max_turns=5,
        compact_fn=compact_fn,
        compaction_threshold_tokens=0,
    )

    compact_fn.assert_not_called()


def test_run_agent_loop_accumulates_cache_token_usage_across_turns() -> None:
    """Cache creation and read token counters must accumulate across loop iterations."""

    client = MagicMock(spec=LLMClient)
    client.chat.side_effect = [
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="1", name="my_tool", arguments={"x": "1"})],
            prompt_tokens=10,
            completion_tokens=5,
            cache_creation_tokens=200,
            cache_read_tokens=0,
        ),
        LLMResponse(
            content="Done",
            tool_calls=[],
            prompt_tokens=15,
            completion_tokens=8,
            cache_creation_tokens=0,
            cache_read_tokens=200,
        ),
    ]

    registry = ToolRegistry()

    @registry.register(
        name="my_tool",
        description="test",
        parameters={"x": {"type": "string", "description": "Input value"}},
    )
    def my_tool(x: str) -> str:
        return f"result: {x}"

    messages: list[dict[str, object]] = [{"role": "user", "content": "Run"}]

    result = run_agent_loop(
        llm_client=client,
        messages=messages,
        tools=registry,
        max_turns=3,
    )

    assert result.cache_creation_tokens == 200
    assert result.cache_read_tokens == 200
