"""Context manager — handles system message injection and tool result truncation.

Two responsibilities:
1. Build the system message by appending a project_state summary to the
   base system prompt (refreshed each turn so core state survives context truncation).
2. Truncate oversized tool results to prevent context bloat, preserving
   head + tail + a truncation notice.
"""

from __future__ import annotations

from .schemas import ProjectState
from .workspace import Workspace


class ContextManager:
    """Manages context injection and truncation for the main agent."""

    def __init__(
        self,
        workspace: Workspace,
        max_tool_result_length: int = 10_000,
    ) -> None:
        """
        Args:
            workspace: Workspace instance for reading project_state.
            max_tool_result_length: Tool results longer than this (in chars)
                will be truncated with head + tail preservation.
        """
        self.workspace = workspace
        self.max_tool_result_length = max_tool_result_length

    def build_system_message(self, base_prompt: str) -> str:
        """Build a system message with project state summary appended.

        Reads the current ProjectState from workspace and appends a
        compact summary to base_prompt. The summary includes:
        - current_stage_id
        - Per-stage status (plan_status, plan_reviewed, stage_status, task count, has_conclusion)
        - data_catalog.json pointer (not its content)

        This ensures core project state is visible to the LLM even if
        earlier conversation messages are truncated by the context window.

        Args:
            base_prompt: The base system prompt template.

        Returns:
            base_prompt + "\\n\\n## Current Project State\\n..." summary.
        """
        state = self.workspace.get_state()
        summary = self._build_project_state_summary(state)
        return f"{base_prompt}\n\n{summary}"

    def truncate_tool_result(self, content: str) -> str:
        """Truncate tool result if it exceeds max_tool_result_length.

        Preserves the first and last portions of the content with a
        truncation notice in between, so the LLM has both the beginning
        (usually headers/structure) and end (usually results/summary).

        Returns content unchanged if within limit.
        """
        if len(content) <= self.max_tool_result_length:
            return content

        head_size = int(self.max_tool_result_length * 0.6)
        tail_size = int(self.max_tool_result_length * 0.2)
        removed_char_count = len(content) - head_size - tail_size
        notice = (
            "\n\n"
            f"[... truncated {removed_char_count} chars; original length {len(content)} chars ...]"
            "\n\n"
        )
        return f"{content[:head_size]}{notice}{content[-tail_size:]}"

    def _build_project_state_summary(self, state: ProjectState) -> str:
        """Render a compact, deterministic text summary of the current project state."""

        if not state.stages:
            return "## Current Project State\nNo stages yet."

        summary_lines = [
            "## Current Project State",
            f"Current stage: {state.current_stage_id or 'None'}",
            "Data catalog: data_catalog.json",
            "Stages:",
        ]

        for stage_id in sorted(state.stages):
            stage = state.stages[stage_id]
            summary_lines.append(
                "- "
                f"{stage.stage_id} | "
                f"plan_status={stage.plan_status or 'None'} | "
                f"plan_reviewed={stage.plan_reviewed} | "
                f"stage_status={stage.stage_status} | "
                f"task_count={len(stage.task_ids)} | "
                f"has_conclusion={stage.has_conclusion}"
            )

        return "\n".join(summary_lines)
