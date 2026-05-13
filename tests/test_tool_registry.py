"""Tests for the tool registry."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agent.tool_registry import ToolError, ToolRegistry


def test_register_decorator_adds_tool_to_definitions() -> None:
    """register stores metadata and keeps the original callable unchanged."""

    registry = ToolRegistry()
    parameters = {
        "query": {"type": "string", "description": "Search query"},
    }

    @registry.register(
        name="search_notes",
        description="Search notes by query.",
        parameters=parameters,
    )
    def search_notes(query: str) -> str:
        return f"Found notes for {query}."

    assert search_notes("cells") == "Found notes for cells."
    assert registry.get_definitions() == [
        {
            "type": "function",
            "function": {
                "name": "search_notes",
                "description": "Search notes by query.",
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": ["query"],
                },
            },
        }
    ]


def test_execute_returns_success_result_for_registered_tool() -> None:
    """execute returns the tool result and marks it as non-error on success."""

    registry = ToolRegistry()

    @registry.register(
        name="concat",
        description="Concatenate two strings.",
        parameters={
            "left": {"type": "string", "description": "Left value"},
            "right": {"type": "string", "description": "Right value"},
        },
    )
    def concat(left: str, right: str) -> str:
        return left + right

    assert registry.execute("concat", {"left": "bio", "right": "logy"}) == ("biology", False)


def test_execute_rejects_hallucinated_extra_arguments() -> None:
    """execute returns a ToolError when the LLM passes unsupported keyword arguments."""

    registry = ToolRegistry()

    @registry.register(
        name="concat",
        description="Concatenate two strings.",
        parameters={
            "left": {"type": "string", "description": "Left value"},
            "right": {"type": "string", "description": "Right value"},
        },
    )
    def concat(left: str, right: str) -> str:
        return left + right

    result, is_error = registry.execute(
        "concat",
        {"left": "bio", "right": "logy", "separator": "-"},
    )
    assert is_error is True
    assert "separator" in result
    assert "unexpected keyword argument" in result


def test_execute_rejects_missing_required_arguments() -> None:
    """execute returns a ToolError when the LLM omits required arguments."""

    registry = ToolRegistry()

    @registry.register(
        name="concat",
        description="Concatenate two strings.",
        parameters={
            "left": {"type": "string", "description": "Left value"},
            "right": {"type": "string", "description": "Right value"},
        },
    )
    def concat(left: str, right: str) -> str:
        return left + right

    result, is_error = registry.execute("concat", {"left": "bio"})

    assert is_error is True
    assert "missing a required argument" in result


def test_execute_returns_error_for_unknown_tool() -> None:
    """execute reports unknown tools as tool-call errors."""

    registry = ToolRegistry()

    assert registry.execute("missing_tool", {"value": "x"}) == ("Unknown tool: missing_tool", True)


def test_execute_catches_tool_error() -> None:
    """execute converts ToolError into an error tool result."""

    registry = ToolRegistry()

    @registry.register(
        name="load_stage",
        description="Load stage data.",
        parameters={
            "stage_id": {"type": "string", "description": "Stage identifier"},
        },
    )
    def load_stage(stage_id: str) -> str:
        raise ToolError(f"Stage not found: {stage_id}")

    assert registry.execute("load_stage", {"stage_id": "stage_99"}) == (
        "Stage not found: stage_99",
        True,
    )


def test_execute_does_not_catch_non_tool_error() -> None:
    """execute lets unexpected exceptions propagate."""

    registry = ToolRegistry()

    @registry.register(
        name="explode",
        description="Raise a bug-like exception.",
        parameters={
            "value": {"type": "string", "description": "Input value"},
        },
    )
    def explode(value: str) -> str:
        raise ValueError(f"Unexpected value: {value}")

    with pytest.raises(ValueError, match="Unexpected value: bad"):
        registry.execute("explode", {"value": "bad"})


def test_get_definitions_returns_openai_function_calling_format() -> None:
    """get_definitions emits the exact OpenAI function-calling structure."""

    registry = ToolRegistry()

    @registry.register(
        name="summarize_paper",
        description="Summarize a paper abstract.",
        parameters={
            "title": {"type": "string", "description": "Paper title"},
            "max_words": {"type": "integer", "description": "Summary length cap"},
        },
    )
    def summarize_paper(title: str, max_words: int) -> str:
        return f"{title} in {max_words} words."

    definitions = registry.get_definitions()

    assert len(definitions) == 1
    assert definitions[0]["type"] == "function"
    assert definitions[0]["function"] == {
        "name": "summarize_paper",
        "description": "Summarize a paper abstract.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Paper title"},
                "max_words": {"type": "integer", "description": "Summary length cap"},
            },
            "required": ["title", "max_words"],
        },
    }


def test_multiple_tools_can_be_registered() -> None:
    """Multiple registrations coexist and remain independently callable."""

    registry = ToolRegistry()

    @registry.register(
        name="uppercase",
        description="Uppercase text.",
        parameters={
            "text": {"type": "string", "description": "Text to uppercase"},
        },
    )
    def uppercase(text: str) -> str:
        return text.upper()

    @registry.register(
        name="repeat",
        description="Repeat text.",
        parameters={
            "text": {"type": "string", "description": "Text to repeat"},
            "times": {"type": "integer", "description": "Repeat count"},
        },
    )
    def repeat(text: str, times: int) -> str:
        return text * times

    definitions = registry.get_definitions()

    assert [definition["function"]["name"] for definition in definitions] == ["uppercase", "repeat"]
    assert registry.execute("uppercase", {"text": "rna"}) == ("RNA", False)
    assert registry.execute("repeat", {"text": "ab", "times": 3}) == ("ababab", False)


def test_get_definitions_marks_only_non_default_parameters_as_required() -> None:
    """Optional handler parameters should not appear in the required schema list."""

    registry = ToolRegistry()

    @registry.register(
        name="dispatch_like",
        description="Dispatch a task.",
        parameters={
            "task_id": {"type": "string", "description": "Task identifier"},
            "role": {"type": "string", "description": "Role name"},
            "max_turns": {"type": "integer", "description": "Turn limit"},
        },
    )
    def dispatch_like(task_id: str, role: str = "general", max_turns: int = 30) -> str:
        return f"{task_id}:{role}:{max_turns}"

    definitions = registry.get_definitions()

    assert definitions[0]["function"]["parameters"]["required"] == ["task_id"]
