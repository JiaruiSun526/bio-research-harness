"""Tests for harness.run()/resume() resume_history preservation (2026-04-18 fix).

Verifies:
  1. Fresh run() produces resume_history with exactly one entry and correct
     counters.
  2. resume() carries over prior subagent_runs (they must survive the reset).
  3. resume() appends a new entry to resume_history (segment_count grows).
  4. resume() archives the prior session.json to sessions/segment_{N}.json.
  5. Pre-fix sessions (no resume_history, only run_result) are migrated to a
     synthetic segment_0 entry tagged migrated_from_pre_fix=True.
  6. _save_crash_session appends a segment with stop_reason='error' so
     aggregate metrics don't silently drop crashed segments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

from conftest import build_stub_probe_result

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agent.harness import ResearchHarness
from research_agent.schemas import AgentLoopResult, TaskResult, TaskSpec
from research_agent.workspace import Workspace


class _ScriptedUser:
    """Feeds prescripted replies so resume doesn't need an LLM."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self._replies = list(replies or ["Continue."])

    def respond(self, message: str, context: object | None = None) -> str:
        del message, context
        if not self._replies:
            return "Acknowledged."
        return self._replies.pop(0)


def _make_harness(tmp_path: Path, *, max_turns: int = 9) -> tuple[ResearchHarness, Workspace]:
    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()
    harness = ResearchHarness(
        model="test-model",
        workspace=workspace,
        user_agent=_ScriptedUser(),
        max_turns=max_turns,
        probe_result=build_stub_probe_result(),
    )
    return harness, workspace


def _approve_stage(workspace: Workspace, stage_id: str) -> None:
    workspace.write_plan(stage_id, "# Plan")
    workspace.mark_plan_reviewed(stage_id)
    workspace.approve_plan(stage_id)


# ---------------------------------------------------------------- fresh run


def test_fresh_run_appends_single_resume_history_entry(tmp_path: Path) -> None:
    harness, workspace = _make_harness(tmp_path)
    _approve_stage(workspace, "stage_01")
    workspace.write_conclusion("stage_01", "# Done")

    with patch("research_agent.harness.run_agent_loop") as mock_loop:
        mock_loop.return_value = AgentLoopResult(
            stop_reason="completed",
            tool_call_count=7,
            turn_count=4,
            prompt_tokens=100,
            completion_tokens=50,
        )
        harness.run()

    session = json.loads((workspace.root / "session.json").read_text())
    assert len(session["resume_history"]) == 1
    entry = session["resume_history"][0]
    assert entry["segment_id"] == 0
    assert entry["stop_reason"] == "completed"
    assert entry["turn_count"] == 4
    assert entry["tool_call_count"] == 7
    assert entry["prompt_tokens"] == 100
    assert entry["completion_tokens"] == 50


# ---------------------------------------------------------------- resume path


def test_resume_preserves_old_subagent_runs(tmp_path: Path) -> None:
    """subagent_runs written by segment 0 must survive through segment 1."""

    harness, workspace = _make_harness(tmp_path)
    _approve_stage(workspace, "stage_01")

    # ---- Segment 0: fresh run that spawns a subagent ----
    def first_loop(**kwargs: object) -> AgentLoopResult:
        task_spec = TaskSpec(
            task_id="task_001",
            stage_id="stage_01",
            task_description="do work",
            max_turns=3,
        )
        # Simulate a sub-agent completion path without invoking a real loop.
        task_result = TaskResult(
            task_id="task_001",
            status="success",
            summary="done",
            artifact_paths=[],
        )
        harness._session_log.record_subagent_run(
            task_spec=task_spec,
            messages=[{"role": "assistant", "content": "ok"}],
            task_result=task_result,
        )
        return AgentLoopResult(stop_reason="max_turns_reached", tool_call_count=1, turn_count=1)

    with patch("research_agent.harness.run_agent_loop", side_effect=first_loop):
        harness.run()

    after_first = json.loads((workspace.root / "session.json").read_text())
    assert len(after_first["subagent_runs"]) == 1

    # ---- Segment 1: resume and verify the old subagent_run survives ----
    harness2, _ = _make_harness(tmp_path, max_turns=3)
    # Point the new harness at the SAME workspace directory.
    harness2.workspace = workspace

    def second_loop(**kwargs: object) -> AgentLoopResult:
        # verify that the resumed session_log already has the prior subagent_run
        assert len(harness2._session_log.subagent_runs) == 1
        return AgentLoopResult(stop_reason="completed", turn_count=2, tool_call_count=3)

    with patch("research_agent.harness.run_agent_loop", side_effect=second_loop):
        harness2.resume(workspace.root)

    after_resume = json.loads((workspace.root / "session.json").read_text())
    # Old subagent_run preserved + (no new ones in this test)
    assert len(after_resume["subagent_runs"]) == 1
    # Two segments recorded now
    assert len(after_resume["resume_history"]) == 2
    assert after_resume["resume_history"][0]["segment_id"] == 0
    assert after_resume["resume_history"][1]["segment_id"] == 1
    assert after_resume["resume_history"][1]["turn_count"] == 2
    assert after_resume["resume_history"][1]["tool_call_count"] == 3


def test_resume_archives_prior_session_json(tmp_path: Path) -> None:
    harness, workspace = _make_harness(tmp_path)
    _approve_stage(workspace, "stage_01")

    with patch("research_agent.harness.run_agent_loop") as mock_loop:
        mock_loop.return_value = AgentLoopResult(stop_reason="max_turns_reached", turn_count=1)
        harness.run()

    harness2, _ = _make_harness(tmp_path)
    harness2.workspace = workspace

    with patch("research_agent.harness.run_agent_loop") as mock_loop2:
        mock_loop2.return_value = AgentLoopResult(stop_reason="completed", turn_count=1)
        harness2.resume(workspace.root)

    archive = workspace.root / "sessions" / "segment_0.json"
    assert archive.is_file(), "prior session.json not archived"
    archived = json.loads(archive.read_text())
    assert archived["resume_history"][0]["stop_reason"] == "max_turns_reached"


# ---------------------------------------------------------------- migration


def test_resume_migrates_pre_fix_session_without_resume_history(tmp_path: Path) -> None:
    """Old session.json with only run_result (no resume_history) must be migrated.

    The archive still needs to happen, and the synthesized segment_0 entry
    should carry migrated_from_pre_fix=True.
    """

    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()

    # Hand-craft a pre-fix session.json with run_result but no resume_history.
    pre_fix_session = {
        "run_id": "legacy",
        "started_at": "2026-04-01T00:00:00+00:00",
        "completed_at": "2026-04-01T01:00:00+00:00",
        "model": "test-model",
        "initial_goal": "legacy goal",
        "main_conversation": [],
        "subagent_runs": [],
        "project_state": {"stages": {}},
        "run_result": {
            "stop_reason": "max_turns",
            "turn_count": 20,
            "tool_call_count": 40,
            "escalation_count": 3,
            "prompt_tokens": 5000,
            "completion_tokens": 2000,
        },
    }
    (workspace.root / "session.json").write_text(json.dumps(pre_fix_session))

    harness = ResearchHarness(
        model="test-model",
        workspace=workspace,
        user_agent=_ScriptedUser(),
        max_turns=3,
        probe_result=build_stub_probe_result(),
    )

    with patch("research_agent.harness.run_agent_loop") as mock_loop:
        mock_loop.return_value = AgentLoopResult(stop_reason="completed", turn_count=1)
        harness.resume(workspace.root)

    new_session = json.loads((workspace.root / "session.json").read_text())
    history = new_session["resume_history"]
    assert len(history) == 2
    assert history[0]["segment_id"] == 0
    assert history[0]["migrated_from_pre_fix"] is True
    assert history[0]["turn_count"] == 20
    assert history[0]["tool_call_count"] == 40
    assert history[0]["escalation_count"] == 3
    assert history[1]["segment_id"] == 1


# ---------------------------------------------------------------- crash path


def test_crash_session_appends_segment_to_resume_history(tmp_path: Path) -> None:
    """Crashed runs must still contribute a segment so aggregation isn't lossy."""

    harness, workspace = _make_harness(tmp_path)
    _approve_stage(workspace, "stage_01")

    class _Boom(RuntimeError):
        pass

    def boom_loop(**kwargs: object) -> AgentLoopResult:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        # Add some assistant turns before crashing so counts aren't zero.
        messages.append({"role": "assistant", "content": "thinking", "tool_calls": []})
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            }
        )
        raise _Boom("simulated crash")

    with patch("research_agent.harness.run_agent_loop", side_effect=boom_loop):
        try:
            harness.run()
        except _Boom:
            pass

    session = json.loads((workspace.root / "session.json").read_text())
    assert len(session["resume_history"]) == 1
    entry = session["resume_history"][0]
    assert entry["stop_reason"] == "error"
    # Crash helper extracts counts from messages: 2 assistant turns, 1 tool call.
    assert entry["turn_count"] == 2
    assert entry["tool_call_count"] == 1
