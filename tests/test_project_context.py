"""Tests for the multi-source project context loader."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agent.project_context import (
    DEFAULT_PROJECT_CONTEXT_FILENAMES,
    load_project_context,
)


def test_load_project_context_returns_empty_when_no_search_paths() -> None:
    """No search paths means no context — empty result, never a dangling header."""

    assert load_project_context(search_paths=None) == ""
    assert load_project_context(search_paths=[]) == ""


def test_load_project_context_returns_empty_when_no_files_match(tmp_path: Path) -> None:
    """Search path exists but contains none of the recognized filenames."""

    (tmp_path / "unrelated.txt").write_text("ignored", encoding="utf-8")
    assert load_project_context(search_paths=[tmp_path]) == ""


def test_load_project_context_skips_empty_files(tmp_path: Path) -> None:
    """Empty/whitespace-only context files contribute nothing — return ""."""

    (tmp_path / "PROJECT_RULES.md").write_text("   \n\n  \n", encoding="utf-8")
    assert load_project_context(search_paths=[tmp_path]) == ""


def test_load_project_context_renders_single_file_with_header(tmp_path: Path) -> None:
    """One non-empty file produces the section header + a per-file subsection."""

    rule_path = tmp_path / "PROJECT_RULES.md"
    rule_path.write_text("Use snake_case for filenames.\n", encoding="utf-8")

    result = load_project_context(search_paths=[tmp_path])

    assert result.startswith("## Project Context\n\n")
    assert f"### From `{rule_path}`" in result
    assert "Use snake_case for filenames." in result


def test_load_project_context_includes_all_recognized_filenames_in_one_path(
    tmp_path: Path,
) -> None:
    """All three recognized filenames in one directory are concatenated in priority order."""

    (tmp_path / "PROJECT_RULES.md").write_text("rules content", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("agents content", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("claude content", encoding="utf-8")

    result = load_project_context(search_paths=[tmp_path])

    rules_pos = result.find("rules content")
    agents_pos = result.find("agents content")
    claude_pos = result.find("claude content")

    # Filename priority: PROJECT_RULES.md → AGENTS.md → CLAUDE.md.
    assert 0 < rules_pos < agents_pos < claude_pos


def test_load_project_context_orders_search_paths_first_to_last(tmp_path: Path) -> None:
    """Earlier search paths appear before later ones in the merged output."""

    repo = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    repo.mkdir()
    workspace.mkdir()
    (repo / "AGENTS.md").write_text("repo level rules", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("workspace level rules", encoding="utf-8")

    result = load_project_context(search_paths=[repo, workspace])

    repo_pos = result.find("repo level rules")
    workspace_pos = result.find("workspace level rules")
    assert 0 < repo_pos < workspace_pos


def test_load_project_context_respects_custom_filenames(tmp_path: Path) -> None:
    """Caller-supplied filenames override the defaults; missing defaults are skipped."""

    (tmp_path / "PROJECT_RULES.md").write_text("default file", encoding="utf-8")
    (tmp_path / "MY_RULES.md").write_text("custom file", encoding="utf-8")

    result = load_project_context(
        search_paths=[tmp_path], filenames=("MY_RULES.md",)
    )

    assert "custom file" in result
    assert "default file" not in result


def test_load_project_context_deduplicates_when_same_path_appears_twice(
    tmp_path: Path,
) -> None:
    """Repeating a path must not duplicate its content in the merged output."""

    (tmp_path / "PROJECT_RULES.md").write_text("only once", encoding="utf-8")

    result = load_project_context(search_paths=[tmp_path, tmp_path])

    assert result.count("only once") == 1


def test_default_filenames_constant_includes_known_aliases() -> None:
    """The default filename tuple is a public part of the contract — assert membership."""

    assert "PROJECT_RULES.md" in DEFAULT_PROJECT_CONTEXT_FILENAMES
    assert "AGENTS.md" in DEFAULT_PROJECT_CONTEXT_FILENAMES
    assert "CLAUDE.md" in DEFAULT_PROJECT_CONTEXT_FILENAMES
