"""Tests for the context manager."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agent.context import ContextManager
from research_agent.workspace import Workspace


def make_workspace(tmp_path: Path) -> Workspace:
    """Create and initialize a workspace rooted under pytest's tmp_path."""

    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()
    return workspace


def test_build_system_message_returns_base_prompt_unchanged(tmp_path: Path) -> None:
    """The stable system message must be byte-identical to base_prompt.

    Project state is no longer appended here — it goes through
    build_state_overlay() so the system prefix can serve as a cache anchor.
    """

    workspace = make_workspace(tmp_path)
    manager = ContextManager(workspace=workspace)

    assert manager.build_system_message("Base prompt") == "Base prompt"


def test_build_system_message_byte_stable_across_state_changes(tmp_path: Path) -> None:
    """Mutating workspace state must NOT change the system message.

    This is the cache-anchor invariant: any drift between turns invalidates
    the provider-side prompt cache.
    """

    workspace = make_workspace(tmp_path)
    manager = ContextManager(workspace=workspace)
    before = manager.build_system_message("Base prompt")

    workspace.write_plan("stage_01", "# Stage 01")
    workspace.mark_plan_reviewed("stage_01")
    workspace.approve_plan("stage_01")

    after = manager.build_system_message("Base prompt")
    assert before == after


def test_build_system_message_appends_project_context_when_provided(
    tmp_path: Path,
) -> None:
    """Non-empty project_context is appended verbatim to the base prompt."""

    workspace = make_workspace(tmp_path)
    project_context = "## Project Context\n\nRule: prefer matplotlib."
    manager = ContextManager(
        workspace=workspace, project_context=project_context
    )

    result = manager.build_system_message("Base prompt")

    assert result == f"Base prompt\n\n{project_context}"


def test_build_system_message_omits_project_context_block_when_empty(
    tmp_path: Path,
) -> None:
    """Empty project_context must not introduce trailing whitespace or headers."""

    workspace = make_workspace(tmp_path)
    manager = ContextManager(workspace=workspace, project_context="")
    assert manager.build_system_message("Base prompt") == "Base prompt"


def test_build_state_overlay_with_empty_project_state(tmp_path: Path) -> None:
    """Empty state should produce the minimal project-state summary."""

    workspace = make_workspace(tmp_path)
    manager = ContextManager(workspace=workspace)

    overlay = manager.build_state_overlay()

    assert overlay == "## Current Project State\nNo stages yet."


def test_build_state_overlay_with_stages_includes_stage_summaries(tmp_path: Path) -> None:
    """Non-empty state should include a compact summary for each stage."""

    workspace = make_workspace(tmp_path)
    workspace.write_plan("stage_01", "# Stage 01")
    workspace.mark_plan_reviewed("stage_01")
    workspace.approve_plan("stage_01")
    workspace.register_task("stage_01", "task_001")
    workspace.write_conclusion("stage_01", "# Conclusion")
    workspace.write_plan("stage_02", "# Stage 02")

    manager = ContextManager(workspace=workspace)

    overlay = manager.build_state_overlay()

    assert overlay.startswith("## Current Project State\n")
    assert "Current stage: stage_02" in overlay
    assert "Data catalog: data_catalog.json" in overlay
    assert (
        "- stage_01 | plan_status=approved | plan_reviewed=True | stage_status=executing | "
        "task_count=1 | has_conclusion=True"
    ) in overlay
    assert (
        "- stage_02 | plan_status=drafting | plan_reviewed=False | stage_status=planning | "
        "task_count=0 | has_conclusion=False"
    ) in overlay


def test_truncate_tool_result_returns_content_unchanged_when_under_limit(tmp_path: Path) -> None:
    """Short content should be returned exactly as-is."""

    workspace = make_workspace(tmp_path)
    manager = ContextManager(workspace=workspace, max_tool_result_length=20)

    assert manager.truncate_tool_result("short content") == "short content"


def test_truncate_tool_result_preserves_head_and_tail_with_notice(tmp_path: Path) -> None:
    """Long content should keep the configured head and tail segments around a notice."""

    workspace = make_workspace(tmp_path)
    manager = ContextManager(workspace=workspace, max_tool_result_length=10)
    content = "abcdefghijklmnopqrstuvwxyz"

    truncated = manager.truncate_tool_result(content)

    assert truncated.startswith("abcdef")
    assert truncated.endswith("yz")
    assert "[... truncated 18 chars; original length 26 chars ...]" in truncated


def test_truncation_notice_includes_character_count(tmp_path: Path) -> None:
    """The truncation notice should expose the original content length."""

    workspace = make_workspace(tmp_path)
    manager = ContextManager(workspace=workspace, max_tool_result_length=15)
    content = "x" * 40

    truncated = manager.truncate_tool_result(content)

    assert "original length 40 chars" in truncated
