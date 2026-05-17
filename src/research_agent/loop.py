"""Core agentic loop — shared by main agent and sub agents.

The loop is role-agnostic: the distinction between main agent and sub agent
is entirely determined by the tools and system prompt passed in. The loop
itself only handles: LLM call → tool dispatch → result append → repeat.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .llm_client import LLMClient
from .schemas import AgentLoopResult
from .schemas import StopReason
from .tool_registry import ToolRegistry


def run_agent_loop(
    *,
    llm_client: LLMClient,
    messages: list[dict[str, Any]],
    tools: ToolRegistry,
    max_turns: int,
    model: str | None = None,
    truncate_fn: Callable[[str], str] | None = None,
    state_overlay_fn: Callable[[], str] | None = None,
    checkpoint_fn: Callable[[], None] | None = None,
    compact_fn: Callable[[], bool] | None = None,
    compaction_threshold_tokens: int = 0,
    trap_errors: bool = False,
    require_terminal_tool: bool = False,
) -> AgentLoopResult:
    """Run the agentic loop until completion, turn limit, or error.

    Loop logic:
        while True:
            request_messages = messages + [{role:user, content: state_overlay}]  # transient
            response = llm_client.chat(request_messages, tools.get_definitions(), model)
            if no tool_calls → break (completed)
            for each tool_call:
                result = tools.execute(call.name, call.arguments)
                result = truncate_fn(result) if truncate_fn else result
                messages.append(tool_result_message)
            turn_count += 1
            if turn_count >= max_turns → break (max_turns_reached)

    State injection model: ``state_overlay_fn`` returns a fresh project-state
    summary string per turn. The loop appends it as a transient user-role
    message *only* for the LLM call, never persists it into ``messages``.
    This keeps the system prefix byte-stable across turns (cache-friendly)
    while still showing the agent up-to-date state on every call.

    Tool errors are handled at two layers:
    - ToolError raised by a tool implementation is caught inside
      tools.execute() and returned to the LLM as an error tool result.
    - Any other exception (TypeError, AttributeError, programming bugs)
      propagates out of this loop. The caller is responsible for any
      cleanup; per project policy, programming bugs should crash loudly.

    The messages list is mutated in place — caller retains access to
    the full conversation history after the loop exits.

    Args:
        llm_client: LLM client for making chat completions.
        messages: Initial message list (including system message).
            Mutated in place as the loop appends assistant and tool messages.
        tools: Tool registry for this agent (main or sub).
        max_turns: Maximum number of LLM calls before forced stop.
        model: Override llm_client's default model.
        truncate_fn: Optional function to truncate oversized tool results.
            Applied to each tool result string before appending to messages.
        state_overlay_fn: Optional function returning a transient state
            summary appended (as a user-role message) to the request *only*
            for the current LLM call. Not persisted into ``messages`` so the
            cached system prefix stays stable.
        checkpoint_fn: Optional callback invoked after each turn completes.
            Used by the harness to incrementally save session.json so that
            conversation state survives process kills.
        trap_errors: When True, convert unexpected exceptions into an
            AgentLoopResult(stop_reason="error") instead of propagating.
            Use this for sub-agent isolation; keep False for main-agent
            crashes that should fail loudly during development.

    Returns:
        AgentLoopResult with stop_reason, final response text, and statistics.
    """
    turn_count = 0
    tool_call_count = 0
    final_response: str | None = None
    prompt_tokens = 0
    completion_tokens = 0
    cache_creation_tokens = 0
    cache_read_tokens = 0
    stop_reason: StopReason = "completed"

    try:
        while True:
            if state_overlay_fn is not None:
                overlay = state_overlay_fn()
                request_messages = list(messages) + [
                    {"role": "user", "content": overlay}
                ]
            else:
                request_messages = messages

            response = llm_client.chat(request_messages, tools.get_definitions(), model)
            prompt_tokens += response.prompt_tokens
            completion_tokens += response.completion_tokens
            cache_creation_tokens += response.cache_creation_tokens
            cache_read_tokens += response.cache_read_tokens

            if response.content is not None:
                final_response = response.content

            if not response.tool_calls:
                if require_terminal_tool:
                    stop_reason = "error"
                    final_response = (
                        "Model returned plain-text completion without calling the terminal tool."
                    )
                    break
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content,
                    }
                )
                stop_reason = "completed"
                break

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }
                    for tool_call in response.tool_calls
                ],
            }
            messages.append(assistant_message)

            terminal_tool_called = False
            for tool_call in response.tool_calls:
                execution_result = tools.execute_with_metadata(tool_call.name, tool_call.arguments)
                result_content = execution_result.content
                if truncate_fn is not None:
                    result_content = truncate_fn(result_content)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_content,
                    }
                )
                tool_call_count += 1

                # Detect user-initiated finish signal from escalate_to_user.
                # The SimulatedUser includes [FINISH] when all research goals
                # are met; the tool handler prefixes with [FINISH_SIGNAL].
                if result_content.startswith("[FINISH_SIGNAL]"):
                    final_response = result_content.removeprefix("[FINISH_SIGNAL] ")
                    stop_reason = "completed"
                    terminal_tool_called = True
                    continue

                if execution_result.is_terminal and not execution_result.is_error:
                    terminal_tool_called = True
                    final_response = result_content

            if terminal_tool_called:
                turn_count += 1
                stop_reason = "completed"
                if checkpoint_fn is not None:
                    checkpoint_fn()
                break

            turn_count += 1
            if checkpoint_fn is not None:
                checkpoint_fn()
            if turn_count >= max_turns:
                stop_reason = "max_turns_reached"
                break

            # Compact *after* the turn but before the next LLM call so the
            # summary feeds into the next request. We use the most recent
            # response's prompt_tokens (≈ current persistent context size)
            # as the trigger signal.
            if (
                compact_fn is not None
                and compaction_threshold_tokens > 0
                and response.prompt_tokens > compaction_threshold_tokens
            ):
                compact_fn()
                if checkpoint_fn is not None:
                    checkpoint_fn()
    except Exception as error:
        if not trap_errors:
            raise
        stop_reason = "error"
        final_response = f"{type(error).__name__}: {error}"

    return AgentLoopResult(
        stop_reason=stop_reason,
        final_response=final_response,
        error_message=final_response if stop_reason == "error" else None,
        tool_call_count=tool_call_count,
        turn_count=turn_count,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
    )
