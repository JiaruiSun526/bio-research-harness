"""Tests for viewer helpers that do not require a real Streamlit runtime."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _load_viewer_module() -> object:
    """Import research_agent.viewer with a lightweight fake streamlit module."""

    fake_streamlit = types.SimpleNamespace()
    sys.modules.setdefault("streamlit", fake_streamlit)
    return importlib.import_module("research_agent.viewer")


def test_get_tool_icon_includes_finish_run() -> None:
    """finish_run should render with a dedicated icon in the viewer."""

    viewer = _load_viewer_module()

    assert viewer._get_tool_icon("finish_run") == "[finish]"  # type: ignore[attr-defined]


def test_format_args_preview_handles_non_dict_payloads() -> None:
    """Argument preview formatting should not assume dict input."""

    viewer = _load_viewer_module()

    preview = viewer._format_args_preview(["bad", "payload"])  # type: ignore[attr-defined]

    assert preview == '["bad", "payload"]'


def test_render_stage_progression_uses_project_state_statuses() -> None:
    """Stage progression should read per-stage status from session project_state."""

    viewer = _load_viewer_module()
    sidebar_calls: list[str] = []
    fake_sidebar = types.SimpleNamespace(
        markdown=lambda text: sidebar_calls.append(text),
        title=lambda *args, **kwargs: None,
        subheader=lambda *args, **kwargs: None,
        checkbox=lambda *args, **kwargs: True,
    )
    viewer.st = types.SimpleNamespace(sidebar=fake_sidebar)  # type: ignore[attr-defined]

    viewer._render_stage_progression(  # type: ignore[attr-defined]
        {
            "project_state": {
                "stages": {
                    "stage_01": {"stage_status": "completed", "task_ids": ["task_001"]},
                    "stage_03": {"stage_status": "executing", "task_ids": ["task_004", "task_005"]},
                }
            }
        }
    )

    assert sidebar_calls == [
        "- **stage_01** (completed) — 1 task(s)",
        "- **stage_03** (executing) — 2 task(s)",
    ]
