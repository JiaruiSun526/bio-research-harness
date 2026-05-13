"""Session log — captures run data for post-hoc review and visualization.

Accumulates events in memory during execution, then writes a single
structured JSON file (session.json) to the workspace after the run completes.
Both tests and the Streamlit viewer consume this format.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import ProjectState, RunResult, TaskResult, TaskSpec


class SessionLog:
    """Captures run data for post-hoc review and visualization.

    Usage by harness:
    1. Create at the start of run() with model and initial_goal.
    2. After _dispatch_subagent(), call record_subagent_run().
    3. After run_agent_loop() returns, call record_main_conversation() and record_run_result().
    4. Call save() to write session.json to workspace.
    """

    def __init__(self, model: str, initial_goal: str) -> None:
        self.run_id = str(uuid.uuid4())
        self.started_at = datetime.now(tz=timezone.utc)
        self.completed_at: datetime | None = None
        self.model = model
        self.initial_goal = initial_goal
        self.main_conversation: list[dict[str, Any]] = []
        self.subagent_runs: list[dict[str, Any]] = []
        self.run_result: dict[str, Any] | None = None
        self.project_state: dict[str, Any] | None = None
        self.resume_history: list[dict[str, Any]] = []
        self.resumed_from: dict[str, Any] | None = None

    def record_main_conversation(self, messages: list[dict[str, Any]]) -> None:
        """Snapshot the main agent's full message list after the loop completes."""
        self.main_conversation = _serialize_messages(messages)

    def record_subagent_run(
        self,
        task_spec: TaskSpec,
        messages: list[dict[str, Any]],
        task_result: TaskResult,
    ) -> None:
        """Record a completed subagent run (called from _dispatch_subagent)."""
        self.subagent_runs.append(
            {
                "task_id": task_spec.task_id,
                "stage_id": task_spec.stage_id,
                "role": task_spec.role,
                "task_description": task_spec.task_description,
                "conversation": _serialize_messages(messages),
                "task_result": task_result.model_dump(),
            }
        )

    def record_run_result(self, run_result: RunResult) -> None:
        """Record the final RunResult after the main loop completes."""
        self.completed_at = datetime.now(tz=timezone.utc)
        self.run_result = run_result.model_dump()

    def record_project_state(self, project_state: ProjectState) -> None:
        """Record the final project state snapshot for viewer rendering."""

        snapshot = project_state.model_dump(mode="json")
        snapshot["stages"] = {
            stage_id: {
                **stage.model_dump(mode="json"),
                "stage_status": stage.stage_status,
            }
            for stage_id, stage in project_state.stages.items()
        }
        self.project_state = snapshot

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full session log to a JSON-serializable dict."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "model": self.model,
            "initial_goal": self.initial_goal,
            "run_result": self.run_result,
            "project_state": self.project_state,
            "main_conversation": self.main_conversation,
            "subagent_runs": self.subagent_runs,
            "resume_history": self.resume_history,
            "resumed_from": self.resumed_from,
        }

    def save(self, path: Path) -> None:
        """Write session.json to the given path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")


def _serialize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deep-copy messages, ensuring all values are JSON-serializable."""
    serialized: list[dict[str, Any]] = []
    for msg in messages:
        entry: dict[str, Any] = {"role": msg.get("role", "unknown")}
        if msg.get("content") is not None:
            entry["content"] = msg["content"]
        if msg.get("tool_calls"):
            entry["tool_calls"] = _serialize_tool_calls(msg["tool_calls"])
        if msg.get("tool_call_id"):
            entry["tool_call_id"] = msg["tool_call_id"]
        serialized.append(entry)
    return serialized


def _serialize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize tool_calls for JSON output (arguments may be string or dict)."""
    result: list[dict[str, Any]] = []
    for tc in tool_calls:
        func = tc.get("function", {})
        args = _normalize_tool_arguments(func.get("arguments", {}))
        result.append(
            {
                "id": tc.get("id"),
                "name": func.get("name"),
                "arguments": args,
            }
        )
    return result


def _normalize_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    """Normalize raw tool arguments into a dict without silently losing parse errors."""

    if isinstance(raw_arguments, dict):
        return raw_arguments

    if isinstance(raw_arguments, str):
        try:
            parsed_arguments = json.loads(raw_arguments)
        except (json.JSONDecodeError, TypeError) as error:
            return {
                "_raw_arguments": raw_arguments,
                "_parse_error": f"{type(error).__name__}: {error}",
            }
        if isinstance(parsed_arguments, dict):
            return parsed_arguments
        return {
            "_raw_arguments": raw_arguments,
            "_parse_error": "Tool arguments JSON must decode to an object.",
            "_parsed_type": type(parsed_arguments).__name__,
        }

    return {
        "_raw_arguments": repr(raw_arguments),
        "_parse_error": "Unsupported tool argument payload type.",
        "_parsed_type": type(raw_arguments).__name__,
    }
