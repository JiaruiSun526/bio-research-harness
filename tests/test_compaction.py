"""Tests for threshold-triggered conversation compaction."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agent.compaction import (
    compact_messages,
    find_cut_index,
    render_history_for_summary,
    summarize_history,
)
from research_agent.llm_client import LLMClient
from research_agent.schemas import LLMResponse


def _assistant(content: str | None = None, tool_calls: list[dict] | None = None) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return msg


def _tool(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _build_three_turn_conversation() -> list[dict]:
    """Build a system + user + 3 assistant turns conversation.

    Each turn has one assistant message + one tool result, so the
    structure is: [system, user, A1, T1, A2, T2, A3, T3].
    """

    return [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "Goal"},
        _assistant(
            content="Turn 1",
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "do", "arguments": json.dumps({"x": 1})},
                }
            ],
        ),
        _tool("c1", "result 1"),
        _assistant(
            content="Turn 2",
            tool_calls=[
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "do", "arguments": json.dumps({"x": 2})},
                }
            ],
        ),
        _tool("c2", "result 2"),
        _assistant(
            content="Turn 3",
            tool_calls=[
                {
                    "id": "c3",
                    "type": "function",
                    "function": {"name": "do", "arguments": json.dumps({"x": 3})},
                }
            ],
        ),
        _tool("c3", "result 3"),
    ]


def test_find_cut_index_returns_kth_from_last_assistant_index() -> None:
    """find_cut_index should point at the assistant message that begins the kept window."""

    messages = _build_three_turn_conversation()
    # Indices: 0=sys, 1=user, 2=A1, 3=T1, 4=A2, 5=T2, 6=A3, 7=T3.
    # keep_last_k_turns=2 should cut at A2 (index 4): A2 + T2 + A3 + T3 retained.
    assert find_cut_index(messages, keep_last_k_turns=2) == 4


def test_find_cut_index_returns_first_assistant_when_k_equals_total_turns() -> None:
    """When K equals the total assistant turn count, the cut sits at the first assistant."""

    messages = _build_three_turn_conversation()
    assert find_cut_index(messages, keep_last_k_turns=3) == 2


def test_find_cut_index_returns_none_when_fewer_assistant_turns_than_k() -> None:
    """Not enough turns to keep K → None signals 'nothing to compact yet'."""

    messages = _build_three_turn_conversation()
    assert find_cut_index(messages, keep_last_k_turns=4) is None


def test_render_history_for_summary_renders_assistant_tool_and_user_roles() -> None:
    """All three role kinds should render with stable prefixes; system rows are skipped."""

    messages = [
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": "Goal"},
        _assistant(
            content="Working",
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q":"foo"}'},
                }
            ],
        ),
        _tool("c1", "found 3 hits"),
        {"role": "user", "content": "Continue"},
    ]

    rendered = render_history_for_summary(messages)

    assert "[user] Goal" in rendered
    assert "[assistant] Working" in rendered
    assert "-> tool_call search(" in rendered
    assert "[tool_result] found 3 hits" in rendered
    assert "[user] Continue" in rendered
    # System content must not appear in the rendered body.
    assert "ignored" not in rendered


def test_render_history_for_summary_truncates_long_tool_arguments_and_results() -> None:
    """Long tool arg blobs and tool results should be elided to keep summaries lean."""

    long_args = json.dumps({"data": "x" * 1000})
    long_result = "y" * 1000
    messages = [
        _assistant(
            content="busy",
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "noop", "arguments": long_args},
                }
            ],
        ),
        _tool("c1", long_result),
    ]

    rendered = render_history_for_summary(messages)

    # tool_call args ellipsized at 240 chars
    assert "…" in rendered.split("-> tool_call noop(")[1].split(")")[0]
    # tool_result truncated at 600 chars
    assert "…" in rendered.split("[tool_result] ")[1]


def test_summarize_history_calls_llm_and_returns_content() -> None:
    """summarize_history should hand history to the LLM and return the content."""

    client = MagicMock(spec=LLMClient)
    client.chat.return_value = LLMResponse(content="SUMMARY", tool_calls=[])

    result = summarize_history(
        llm_client=client, history_text="[user] Goal\n[assistant] hi"
    )

    assert result == "SUMMARY"
    sent_messages = client.chat.call_args.kwargs["messages"]
    assert sent_messages[0]["role"] == "system"
    assert "summarizing" in sent_messages[0]["content"].lower()
    assert "[user] Goal" in sent_messages[1]["content"]


def test_compact_messages_returns_false_when_fewer_than_three_messages() -> None:
    """A trivially short conversation cannot be compacted."""

    client = MagicMock(spec=LLMClient)
    messages = [{"role": "system", "content": "Sys"}]
    assert compact_messages(messages, llm_client=client, keep_last_k_turns=2) is False
    client.chat.assert_not_called()


def test_compact_messages_returns_false_when_not_enough_turns_to_keep() -> None:
    """When find_cut_index returns None, no LLM call should be made."""

    client = MagicMock(spec=LLMClient)
    messages = _build_three_turn_conversation()
    assert (
        compact_messages(messages, llm_client=client, keep_last_k_turns=10) is False
    )
    client.chat.assert_not_called()


def test_compact_messages_replaces_old_span_with_summary_user_message() -> None:
    """A successful compaction folds [system, user, ...older...] → [system, user, summary, ...kept...].

    The first two messages (system + initial user goal) and the most recent
    K turns (assistant + their tool rows) must remain untouched. A single
    user-role message containing the LLM-produced summary takes the place
    of the older span at index 2.
    """

    client = MagicMock(spec=LLMClient)
    client.chat.return_value = LLMResponse(content="EARLIER SUMMARY", tool_calls=[])
    messages = _build_three_turn_conversation()

    changed = compact_messages(messages, llm_client=client, keep_last_k_turns=1)

    assert changed is True
    # Prefix preserved verbatim.
    assert messages[0] == {"role": "system", "content": "Sys"}
    assert messages[1] == {"role": "user", "content": "Goal"}
    # Single replacement message at index 2 with the LLM summary.
    assert messages[2]["role"] == "user"
    assert "EARLIER SUMMARY" in messages[2]["content"]
    assert "## Summary of Earlier Turns" in messages[2]["content"]
    # Last turn (A3 + T3) preserved verbatim with intact tool_call_id pairing.
    assert messages[3]["role"] == "assistant"
    assert messages[3]["content"] == "Turn 3"
    assert messages[3]["tool_calls"][0]["id"] == "c3"
    assert messages[4] == _tool("c3", "result 3")
    # Total length: 2 (prefix) + 1 (summary) + 2 (last turn) = 5.
    assert len(messages) == 5


def test_compact_messages_preserves_tool_call_pairing_in_kept_window() -> None:
    """For each kept turn, the assistant's tool_call_ids must still be paired with their tool rows.

    If the cut landed inside a turn, the kept window would contain a tool
    result whose matching tool_call_id was just summarized away — providers
    reject that. The test asserts every tool message in the kept window
    matches an assistant tool_call earlier in the window.
    """

    client = MagicMock(spec=LLMClient)
    client.chat.return_value = LLMResponse(content="EARLIER", tool_calls=[])
    messages = _build_three_turn_conversation()

    compact_messages(messages, llm_client=client, keep_last_k_turns=2)

    kept = messages[3:]
    declared_ids: set[str] = set()
    for msg in kept:
        if msg["role"] == "assistant":
            for tc in msg.get("tool_calls", []) or []:
                declared_ids.add(tc["id"])
        elif msg["role"] == "tool":
            assert msg["tool_call_id"] in declared_ids


def test_compact_messages_passes_through_model_override() -> None:
    """The model kwarg should be forwarded to the LLM summarizer call."""

    client = MagicMock(spec=LLMClient)
    client.chat.return_value = LLMResponse(content="ok", tool_calls=[])
    messages = _build_three_turn_conversation()

    compact_messages(
        messages, llm_client=client, keep_last_k_turns=1, model="claude-haiku"
    )

    assert client.chat.call_args.kwargs["model"] == "claude-haiku"
