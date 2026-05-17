"""Threshold-triggered conversation compaction for the main agent.

When the main agent's context grows past a configured token threshold, the
loop calls into ``compact_messages`` to:
1. find a turn-boundary-safe cut point that keeps the most recent K turns,
2. ask the LLM to produce a compact summary of everything before that cut,
3. replace the compacted span (in place) with a single user-role message
   containing the summary.

The system message and the *initial* user message (the research goal) are
always preserved verbatim — only the middle "old turns" portion is
compressed. Tool_call ↔ tool_result pairing is preserved by cutting only at
assistant-message boundaries: each turn begins with an assistant message
that owns the subsequent tool-result rows.

Compaction is for the main agent only. Sub-agent loops are short-lived and
don't need it. The harness wires this in for ``run`` and ``resume``; the
loop is otherwise unaware of how summaries are produced.
"""

from __future__ import annotations

from typing import Any

from .llm_client import LLMClient


_SUMMARY_INSTRUCTION = (
    "You are summarizing a research conversation between a coordinator agent, "
    "its sub-agents, and a user. Produce a concise but information-dense "
    "summary that another instance of the coordinator could use to continue "
    "the work. Cover, in this order:\n"
    "1. Research goal and current focus.\n"
    "2. Stages started and their status (planning / executing / completed).\n"
    "3. Key sub-agent task outcomes — what was produced and where (artifact "
    "paths), what failed and why.\n"
    "4. User decisions, approvals, and pushbacks (preserve any explicit "
    "directives verbatim where short).\n"
    "5. Open questions or pending blockers the agent must resolve next.\n\n"
    "Do not invent facts. Quote artifact paths and stage IDs exactly. Aim for "
    "300–600 words; less is fine if the history is short."
)


def find_cut_index(
    messages: list[dict[str, Any]], keep_last_k_turns: int
) -> int | None:
    """Return the message index where the 'keep last K turns' window begins.

    A *turn* is anchored by an assistant message; the tool result rows that
    immediately follow it belong to the same turn. We walk from the tail
    counting assistant messages — when we have seen K of them, the index of
    that Kth-from-last assistant is the cut. ``messages[cut:]`` is the kept
    portion; ``messages[:cut]`` is the candidate compaction window.

    Returns None when the conversation contains fewer than K assistant turns
    (nothing useful to compact yet).
    """

    if keep_last_k_turns <= 0:
        return len(messages)

    seen = 0
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            seen += 1
            if seen == keep_last_k_turns:
                return i
    return None


def render_history_for_summary(messages: list[dict[str, Any]]) -> str:
    """Render assistant / tool / user messages as plain text for summarization.

    We hand this to the summarizer LLM as a single user message rather than
    replaying the original tool-call structure: replaying the raw structure
    would orphan tool_call_ids relative to the summarizer's empty tool list
    and many providers reject that.
    """

    parts: list[str] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "assistant":
            text = content if isinstance(content, str) else ""
            tool_calls = msg.get("tool_calls") or []
            for call in tool_calls:
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                name = fn.get("name", "?")
                args = fn.get("arguments", "")
                if isinstance(args, str) and len(args) > 240:
                    args = args[:240] + "…"
                text += f"\n  -> tool_call {name}({args})"
            parts.append(f"[assistant] {text.strip()}")
        elif role == "tool":
            tool_text = content if isinstance(content, str) else str(content)
            if len(tool_text) > 600:
                tool_text = tool_text[:600] + "…"
            parts.append(f"[tool_result] {tool_text}")
        elif role == "user":
            parts.append(f"[user] {content}")
        elif role == "system":
            # System content is preserved separately; skip from the body.
            continue
    return "\n".join(parts)


def summarize_history(
    *,
    llm_client: LLMClient,
    history_text: str,
    model: str | None = None,
) -> str:
    """Ask the LLM to produce a compact summary of rendered history."""

    messages = [
        {"role": "system", "content": _SUMMARY_INSTRUCTION},
        {
            "role": "user",
            "content": "Conversation history to summarize:\n\n" + history_text,
        },
    ]
    response = llm_client.chat(messages=messages, model=model)
    return response.content or ""


def compact_messages(
    messages: list[dict[str, Any]],
    *,
    llm_client: LLMClient,
    keep_last_k_turns: int,
    model: str | None = None,
    summary_header: str = "## Summary of Earlier Turns",
) -> bool:
    """Compact ``messages`` in place. Returns True iff compaction happened.

    The first two messages (system + initial user goal) are always kept.
    Indices ``[2:cut)`` are summarized into a single user message and that
    span is replaced by the summary message; ``[cut:]`` (the most recent K
    turns) is preserved verbatim so live tool_call ↔ tool_result references
    stay intact.

    No-op (returns False) when there are fewer than K+1 assistant turns or
    when the cut would land inside the system + initial-user prefix.
    """

    if len(messages) < 3:
        return False
    cut = find_cut_index(messages, keep_last_k_turns)
    if cut is None or cut <= 2:
        return False

    history_text = render_history_for_summary(messages[2:cut])
    summary = summarize_history(
        llm_client=llm_client, history_text=history_text, model=model
    )
    replacement = {
        "role": "user",
        "content": f"{summary_header}\n{summary}",
    }
    messages[2:cut] = [replacement]
    return True
