"""Tests for the workspace filesystem manager."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agent.schemas import TaskResult
from research_agent.workspace import Workspace, WorkspaceError


def make_workspace(tmp_path: Path) -> Workspace:
    """Create and initialize a workspace rooted under pytest's tmp_path."""

    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()
    return workspace


def make_task_result(task_id: str) -> TaskResult:
    """Build a representative task result for round-trip testing."""

    return TaskResult(
        task_id=task_id,
        status="success",
        summary="Task completed.",
        artifact_paths=["stages/stage_01/outputs/report.md"],
        blockers=[],
        suggestions=["Use the report in the final write-up."],
        error=None,
    )


def test_initialize_creates_directory_structure_and_state(tmp_path: Path) -> None:
    """initialize creates the expected layout, state file, and optional data catalog copy."""

    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text('{"datasets": ["paper_a"]}', encoding="utf-8")

    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize(data_catalog_path=catalog_path)
    workspace.initialize(data_catalog_path=catalog_path)

    assert workspace.root.is_dir()
    assert (workspace.root / "plans").is_dir()
    assert (workspace.root / "stages").is_dir()
    assert (workspace.root / "project_state.json").is_file()
    assert (workspace.root / "data_catalog.json").read_text(encoding="utf-8") == (
        '{"datasets": ["paper_a"]}'
    )

    state = workspace.get_state()
    assert state.current_stage_id is None
    assert state.stages == {}


def test_write_plan_creates_file_and_updates_state(tmp_path: Path) -> None:
    """write_plan persists the plan markdown and moves the stage into drafting/planning."""

    workspace = make_workspace(tmp_path)

    workspace.write_plan("stage_01", "# Plan")

    assert (workspace.root / "plans" / "stage_01_plan.md").read_text(encoding="utf-8") == "# Plan"

    state = workspace.get_state()
    stage = state.stages["stage_01"]
    assert state.current_stage_id == "stage_01"
    assert stage.plan_status == "drafting"
    assert stage.stage_status == "planning"
    assert stage.task_ids == []
    assert stage.has_conclusion is False


def test_write_plan_overwrites_existing_drafting_plan(tmp_path: Path) -> None:
    """write_plan supports replanning while the stage is still drafting."""

    workspace = make_workspace(tmp_path)
    workspace.write_plan("stage_01", "# Initial plan")

    workspace.write_plan("stage_01", "# Revised plan")

    assert (workspace.root / "plans" / "stage_01_plan.md").read_text(encoding="utf-8") == (
        "# Revised plan"
    )
    state = workspace.get_state()
    assert state.stages["stage_01"].plan_status == "drafting"
    assert state.stages["stage_01"].plan_reviewed is False
    assert state.stages["stage_01"].stage_status == "planning"


def test_approve_plan_requires_prior_user_review(tmp_path: Path) -> None:
    """approve_plan rejects drafting plans that were never routed through user review."""

    workspace = make_workspace(tmp_path)
    workspace.write_plan("stage_01", "# Plan")

    with pytest.raises(WorkspaceError, match="reviewed with the user"):
        workspace.approve_plan("stage_01")


def test_write_plan_on_approved_plan_raises(tmp_path: Path) -> None:
    """Approved plans are locked and cannot be overwritten."""

    workspace = make_workspace(tmp_path)
    workspace.write_plan("stage_01", "# Plan")
    workspace.mark_plan_reviewed("stage_01")
    workspace.approve_plan("stage_01")

    with pytest.raises(WorkspaceError, match="approved"):
        workspace.write_plan("stage_01", "# New plan")

    assert (workspace.root / "plans" / "stage_01_plan.md").read_text(encoding="utf-8") == "# Plan"


def test_approve_plan_happy_path(tmp_path: Path) -> None:
    """approve_plan transitions a drafted stage into execution."""

    workspace = make_workspace(tmp_path)
    workspace.write_plan("stage_01", "# Plan")
    workspace.mark_plan_reviewed("stage_01")

    workspace.approve_plan("stage_01")

    stage = workspace.get_state().stages["stage_01"]
    assert stage.plan_status == "approved"
    assert stage.stage_status == "executing"


def test_approve_plan_without_plan_raises(tmp_path: Path) -> None:
    """approve_plan requires the plan markdown file to exist."""

    workspace = make_workspace(tmp_path)

    with pytest.raises(WorkspaceError, match="Plan file not found"):
        workspace.approve_plan("stage_01")


def test_approve_plan_on_already_approved_raises(tmp_path: Path) -> None:
    """approve_plan only accepts plans still in drafting status."""

    workspace = make_workspace(tmp_path)
    workspace.write_plan("stage_01", "# Plan")
    workspace.mark_plan_reviewed("stage_01")
    workspace.approve_plan("stage_01")

    with pytest.raises(WorkspaceError, match="drafting"):
        workspace.approve_plan("stage_01")


def test_register_task_happy_path(tmp_path: Path) -> None:
    """register_task appends task ids once the stage plan is approved."""

    workspace = make_workspace(tmp_path)
    workspace.write_plan("stage_01", "# Plan")
    workspace.mark_plan_reviewed("stage_01")
    workspace.approve_plan("stage_01")

    workspace.register_task("stage_01", "task_001")
    workspace.register_task("stage_01", "task_002")

    assert workspace.get_state().stages["stage_01"].task_ids == ["task_001", "task_002"]


def test_register_task_does_not_duplicate_existing_task_id(tmp_path: Path) -> None:
    """register_task is idempotent for retries of the same task identifier."""

    workspace = make_workspace(tmp_path)
    workspace.write_plan("stage_01", "# Plan")
    workspace.mark_plan_reviewed("stage_01")
    workspace.approve_plan("stage_01")

    workspace.register_task("stage_01", "task_001")
    workspace.register_task("stage_01", "task_001")

    assert workspace.get_state().stages["stage_01"].task_ids == ["task_001"]


def test_register_task_requires_approved_plan(tmp_path: Path) -> None:
    """register_task rejects stages that are still only in drafting state."""

    workspace = make_workspace(tmp_path)
    workspace.write_plan("stage_01", "# Plan")

    with pytest.raises(WorkspaceError, match="not approved"):
        workspace.register_task("stage_01", "task_001")


def test_full_dispatch_flow(tmp_path: Path) -> None:
    """The end-to-end stage flow persists files and auto-completes the stage."""

    workspace = make_workspace(tmp_path)

    workspace.write_plan("stage_01", "# Plan")
    workspace.mark_plan_reviewed("stage_01")
    workspace.approve_plan("stage_01")
    workspace.register_task("stage_01", "task_001")

    result = make_task_result("task_001")
    workspace.write_task_result("stage_01", "task_001", result)
    workspace.write_conclusion("stage_01", "# Conclusion")

    result_path = workspace.root / "stages" / "stage_01" / "tasks" / "task_001_result.json"
    conclusion_path = workspace.root / "stages" / "stage_01" / "conclusion.md"
    assert TaskResult.model_validate_json(result_path.read_text(encoding="utf-8")) == result
    assert conclusion_path.read_text(encoding="utf-8") == "# Conclusion"

    state = workspace.get_state()
    assert state.current_stage_id == "stage_01"
    assert state.stages["stage_01"].stage_status == "completed"
    assert state.stages["stage_01"].has_conclusion is True
    assert state.stages["stage_01"].task_ids == ["task_001"]


def test_stage_status_auto_derived(tmp_path: Path) -> None:
    """Stage status is derived from plan approval, task results, and conclusion state."""

    workspace = make_workspace(tmp_path)
    workspace.write_plan("stage_01", "# Plan")
    assert workspace.get_state().stages["stage_01"].stage_status == "planning"

    workspace.mark_plan_reviewed("stage_01")
    workspace.approve_plan("stage_01")
    assert workspace.get_state().stages["stage_01"].stage_status == "executing"

    workspace.register_task("stage_01", "task_001")
    workspace.write_conclusion("stage_01", "# Conclusion")
    assert workspace.get_state().stages["stage_01"].stage_status == "executing"

    workspace.write_task_result("stage_01", "task_001", make_task_result("task_001"))
    assert workspace.get_state().stages["stage_01"].stage_status == "completed"


def test_project_state_does_not_persist_derived_stage_status(tmp_path: Path) -> None:
    """stage_status should be derived on read, not stored as persisted source of truth."""

    workspace = make_workspace(tmp_path)
    workspace.write_plan("stage_01", "# Plan")
    workspace.mark_plan_reviewed("stage_01")
    workspace.approve_plan("stage_01")

    payload = json.loads((workspace.root / "project_state.json").read_text(encoding="utf-8"))

    assert "stage_status" not in payload["stages"]["stage_01"]


def test_read_file_reads_workspace_file(tmp_path: Path) -> None:
    """read_file resolves and reads normal workspace files."""

    workspace = make_workspace(tmp_path)
    workspace.write_plan("stage_01", "# Plan")

    assert workspace.read_file("plans/stage_01_plan.md") == "# Plan"


def test_read_file_rejects_path_traversal(tmp_path: Path) -> None:
    """read_file blocks access to files outside the workspace root."""

    workspace = make_workspace(tmp_path)
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret", encoding="utf-8")

    with pytest.raises(WorkspaceError, match="escapes workspace root"):
        workspace.read_file("../outside.txt")


def test_write_artifact_returns_relative_path(tmp_path: Path) -> None:
    """write_artifact stores the artifact under stage outputs and returns its relative path."""

    workspace = make_workspace(tmp_path)
    workspace.write_plan("stage_01", "# Plan")

    relative_path = workspace.write_artifact("stage_01", "report.md", "# Artifact")

    assert relative_path == "stages/stage_01/outputs/report.md"
    assert (workspace.root / relative_path).read_text(encoding="utf-8") == "# Artifact"


def test_write_task_result_requires_registered_task(tmp_path: Path) -> None:
    """write_task_result should reject task files for unregistered task ids."""

    workspace = make_workspace(tmp_path)
    workspace.write_plan("stage_01", "# Plan")
    workspace.mark_plan_reviewed("stage_01")
    workspace.approve_plan("stage_01")

    with pytest.raises(WorkspaceError, match="not registered"):
        workspace.write_task_result("stage_01", "task_001", make_task_result("task_001"))


def test_write_task_result_rejects_missing_stage(tmp_path: Path) -> None:
    """write_task_result should not create files for stages missing from project state."""

    workspace = make_workspace(tmp_path)

    with pytest.raises(WorkspaceError, match="Stage not found"):
        workspace.write_task_result("stage_missing", "task_001", make_task_result("task_001"))


def test_write_artifact_rejects_missing_stage(tmp_path: Path) -> None:
    """write_artifact should not create output files for stages missing from project state."""

    workspace = make_workspace(tmp_path)

    with pytest.raises(WorkspaceError, match="Stage not found"):
        workspace.write_artifact("stage_missing", "report.md", "# Artifact")
