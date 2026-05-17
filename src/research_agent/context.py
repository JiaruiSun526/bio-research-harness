"""Context manager — system message + state overlay + tool result truncation.

Three responsibilities:
1. Build the *stable* system message (base prompt only) — kept byte-stable across
   turns so it can serve as a prompt-cache anchor.
2. Build the *dynamic* state overlay (current ProjectState summary) — re-rendered
   each turn and injected as a transient user-role message right before the LLM
   call, then dropped from history. This way the cache prefix never changes
   even though the agent always sees fresh state.
3. Truncate oversized tool results, preserving head + tail + a truncation notice.

This stable/dynamic split is required for prompt caching to be effective. The
old design (state appended into messages[0]["content"] each turn) invalidated
the system-message cache on every state change.
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
        project_context: str = "",
    ) -> None:
        """
        Args:
            workspace: Workspace instance for reading project_state.
            max_tool_result_length: Tool results longer than this (in chars)
                will be truncated with head + tail preservation.
            project_context: Optional project-level guidance (PROJECT_RULES.md,
                AGENTS.md, CLAUDE.md, etc.) loaded once at harness init by
                ``project_context.load_project_context``. Appended verbatim to
                the stable system block, so it must not change during the run.
                An empty string (the default) means no project context.
        """
        self.workspace = workspace
        self.max_tool_result_length = max_tool_result_length
        self.project_context = project_context

    def build_system_message(self, base_prompt: str) -> str:
        """Return the stable system message: base prompt + project context.

        Both halves are byte-stable across turns so the provider-side prompt
        cache can reuse the prefix. Dynamic state goes through
        ``build_state_overlay`` instead. Project context is appended only
        when non-empty so the system message stays minimal otherwise.
        """

        if not self.project_context:
            return base_prompt
        return f"{base_prompt}\n\n{self.project_context}"

    def build_state_overlay(self) -> str:
        """Render the current ProjectState as a transient user-message payload.

        This text is appended as a fresh user-role message immediately before
        each LLM call and dropped after the call returns; it is *not* mutated
        into messages[0] (which would invalidate the cache prefix). The agent
        therefore always sees up-to-date project state without paying the
        cache-write cost on every turn.
        """

        state = self.workspace.get_state()
        return self._build_project_state_summary(state)

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
