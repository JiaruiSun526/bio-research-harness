"""Tests for session-log serialization helpers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agent.session_log import _serialize_tool_calls


def test_serialize_tool_calls_preserves_invalid_json_as_parse_error_metadata() -> None:
    """Invalid JSON arguments should be preserved with explicit parse-error metadata."""

    serialized = _serialize_tool_calls(
        [
            {
                "id": "call_1",
                "function": {
                    "name": "finish_run",
                    "arguments": "{not-json}",
                },
            }
        ]
    )

    assert serialized[0]["id"] == "call_1"
    assert serialized[0]["name"] == "finish_run"
    assert serialized[0]["arguments"]["_raw_arguments"] == "{not-json}"
    assert "JSONDecodeError" in serialized[0]["arguments"]["_parse_error"]


def test_serialize_tool_calls_marks_non_object_json_arguments() -> None:
    """JSON that decodes to a non-object should not be silently accepted."""

    serialized = _serialize_tool_calls(
        [
            {
                "id": "call_2",
                "function": {
                    "name": "dispatch_subagent",
                    "arguments": '["not", "an", "object"]',
                },
            }
        ]
    )

    assert serialized == [
        {
            "id": "call_2",
            "name": "dispatch_subagent",
            "arguments": {
                "_raw_arguments": '["not", "an", "object"]',
                "_parse_error": "Tool arguments JSON must decode to an object.",
                "_parsed_type": "list",
            },
        }
    ]
