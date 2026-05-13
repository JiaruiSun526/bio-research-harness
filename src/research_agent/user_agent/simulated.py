"""SimulatedUser — LLM-driven user, guided by a research brief."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..llm_client import LLMClient
from ..schemas import ReviewContext
from ..workspace import WorkspaceError, resolve_path_within_root

logger = logging.getLogger(__name__)


class SimulatedUser:
    """LLM-based simulated user, driven by a research brief.

    Behavior:
    - Research brief is injected as system prompt once at construction,
      and never changes throughout the session.
    - Conversation history across turns is maintained (simulating a
      researcher who remembers prior discussion).
    - When context contains artifact_paths, their file contents are loaded
      into the current LLM call (inspect), but NOT retained across turns —
      simulating a researcher who "reviews but doesn't memorize every line".
    - When the LLM emits [FINISH], a verification LLM call checks whether
      all research goals from the brief are covered by workspace stage
      conclusions. If not, [FINISH] is stripped and missing goals are
      fed back so the user can direct the agent to address them.
    """

    def __init__(
        self,
        research_brief: str,
        llm_client: LLMClient,
        model: str,
        workspace_root: Path,
    ) -> None:
        """
        Args:
            research_brief: Full text of the research brief, used as
                the simulated user's system prompt.
            llm_client: LLM client for generating responses.
            model: Model identifier for simulated user calls.
            workspace_root: Workspace root for resolving artifact paths.
        """
        self.research_brief: str = research_brief
        self.llm_client: LLMClient = llm_client
        self.model: str = model
        self.workspace_root: Path = workspace_root
        self.conversation_history: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a researcher who has read the following research brief. "
                    "You are the human user in this research collaboration.\n\n"
                    "## Behavioral Guidelines\n"
                    "- When the agent presents a plan, review it against your research brief. "
                    "Approve if it addresses your goals; push back with specific changes if not.\n"
                    "- When approving, say 'approved' or 'go ahead' explicitly.\n"
                    "- Keep responses to 2-4 sentences. Be specific and actionable.\n"
                    "- After seeing stage results, decide and state what to do next "
                    "(e.g. 'now run pathway enrichment' or 'let's compare with B cells').\n"
                    "- Do not introduce requirements that aren't in the research brief.\n"
                    "- If results look wrong or methods seem inappropriate, say so.\n\n"
                    "## How to End the Session (CRITICAL)\n"
                    "- The session only ends when you include the literal token `[FINISH]` "
                    "(square brackets, uppercase, exactly as written) in your response.\n"
                    "- Saying 'the project is complete', 'no further action needed', "
                    "'publication-ready', or 'looks good, we're done' in natural language "
                    "does NOT end the session. The agent will keep asking until it sees `[FINISH]`.\n"
                    "- Emit `[FINISH]` as soon as you believe ALL numbered Research Goals in "
                    "your brief are addressed by the agent's completed stage conclusions. "
                    "Do not wait for the agent to ask repeatedly — if work is done, say so and "
                    "include `[FINISH]` in the SAME response.\n"
                    "- If the agent is asking for 'final verification' or 'any further changes' "
                    "and you have already confirmed the work once, either direct new work or "
                    "include `[FINISH]` to terminate. Do not restate the same confirmation twice.\n"
                    "- Do NOT emit `[FINISH]` prematurely: a system check will verify goal "
                    "coverage and reject it if goals remain unaddressed.\n\n"
                    f"## Research Brief\n{research_brief}"
                ),
            }
        ]

    MAX_FINISH_RETRIES = 3
    MAX_EMPTY_RETRIES = 3

    def respond(self, message: str, context: ReviewContext | None = None) -> str:
        """Generate a simulated user response via LLM.

        If context.artifact_paths is provided, reads those files and
        includes their contents in the current LLM call as ephemeral
        context (not retained in conversation history).

        An empty response from the user-agent LLM must never reach the
        main agent: the main agent would then produce a plain-text reply
        with no tool_calls, tripping require_terminal_tool and crashing
        the run. The initial call therefore retries up to MAX_EMPTY_RETRIES
        when content is empty, and raises RuntimeError if every retry
        returns empty.

        When the response contains [FINISH], a verification LLM call
        checks whether all research goals are covered by workspace
        conclusions. If incomplete, the [FINISH] response is discarded
        and the user-agent LLM is re-called with the rejection verdict
        so it produces a natural continuation directing the main agent
        to address missing goals. Retries up to MAX_FINISH_RETRIES; if
        the user agent cannot produce a valid (non-empty, non-[FINISH])
        response, raises RuntimeError so the rejected [FINISH] message
        never reaches the main agent.
        """
        self.conversation_history.append({"role": "user", "content": message})

        ephemeral_artifact_context = self._build_artifact_context(context)

        response_content = self._call_until_nonempty(ephemeral_artifact_context)

        if "[FINISH]" in response_content:
            verdict = self._verify_finish()
            if not verdict.startswith("COMPLETE"):
                response_content = self._regenerate_after_rejection(
                    verdict, ephemeral_artifact_context
                )

        self.conversation_history.append({"role": "assistant", "content": response_content})
        return response_content

    def _call_until_nonempty(self, ephemeral_artifact_context: str | None) -> str:
        """Call the user-agent LLM until it returns non-empty content.

        Raises RuntimeError after MAX_EMPTY_RETRIES consecutive empty
        responses. Empty content from the user agent would otherwise be
        sent back to the main agent as an escalate_to_user result,
        causing the main agent to emit plain text on the next turn and
        trip the require_terminal_tool protocol check.
        """
        for attempt in range(1, self.MAX_EMPTY_RETRIES + 1):
            extra_ephemeral: str | None = None
            if attempt > 1:
                extra_ephemeral = (
                    f"[Retry {attempt - 1}] Your previous response was empty. "
                    "Produce a substantive natural-language reply to the "
                    "agent's last message — either direct the next piece of "
                    "work, or include [FINISH] if all numbered Research Goals "
                    "are now covered."
                )
            call_messages = self._build_call_messages(
                ephemeral_artifact_context, extra_ephemeral=extra_ephemeral
            )
            response = self.llm_client.chat(
                messages=call_messages, model=self.model
            )
            candidate = (response.content or "").strip()
            if candidate:
                if attempt > 1:
                    logger.info(
                        "Simulated user empty-retry succeeded on attempt %d", attempt
                    )
                return candidate
            logger.warning(
                "Simulated user returned empty content (attempt %d/%d)",
                attempt, self.MAX_EMPTY_RETRIES,
            )
        raise RuntimeError(
            "SimulatedUser returned empty content "
            f"{self.MAX_EMPTY_RETRIES} times in a row — refusing to pass "
            "an empty escalate_to_user result back to the main agent."
        )

    def _regenerate_after_rejection(
        self,
        verdict: str,
        ephemeral_artifact_context: str | None,
    ) -> str:
        """Re-call the user-agent LLM until it produces a valid continuation.

        Valid means: non-empty content that does not contain [FINISH]. The
        verdict (rejection reason from _verify_finish) is supplied as
        ephemeral feedback so the user agent knows which goals remain.
        Raises RuntimeError after MAX_FINISH_RETRIES to prevent the
        rejected [FINISH] message or empty content from reaching the main
        agent.
        """
        base_feedback = (
            "You attempted to finish, but not all research goals from your "
            "brief are covered yet. Here is what's missing:\n"
            + verdict
            + "\nRespond to the agent naturally — direct them to address "
            "the missing goals next. Do NOT include [FINISH] in your response."
        )
        for attempt in range(1, self.MAX_FINISH_RETRIES + 1):
            feedback = base_feedback
            if attempt > 1:
                feedback = (
                    base_feedback
                    + f"\n\n[Retry {attempt - 1}] Your previous attempt was "
                    "rejected because it was empty or still contained [FINISH]. "
                    "Produce a substantive natural-language reply telling the "
                    "agent what specific work to do next."
                )
            call_messages = self._build_call_messages(
                ephemeral_artifact_context, extra_ephemeral=feedback
            )
            response = self.llm_client.chat(
                messages=call_messages, model=self.model
            )
            candidate = (response.content or "").strip()
            if candidate and "[FINISH]" not in candidate:
                logger.info(
                    "Simulated user regeneration accepted on attempt %d", attempt
                )
                return candidate
            logger.warning(
                "Simulated user regeneration attempt %d rejected "
                "(empty=%s, contains_finish=%s)",
                attempt,
                not candidate,
                "[FINISH]" in candidate,
            )
        raise RuntimeError(
            "SimulatedUser could not produce a valid non-[FINISH] response "
            f"after {self.MAX_FINISH_RETRIES} retries. Verdict: {verdict}"
        )

    def _build_call_messages(
        self,
        ephemeral_artifact_context: str | None,
        extra_ephemeral: str | None = None,
    ) -> list[dict[str, str]]:
        """Build the message list for an LLM call.

        Starts from conversation_history, then appends ephemeral
        contexts (artifacts, extra nudge) that are NOT persisted.
        """
        call_messages = list(self.conversation_history)
        if ephemeral_artifact_context:
            call_messages.append({"role": "user", "content": ephemeral_artifact_context})
        if extra_ephemeral:
            call_messages.append({"role": "user", "content": extra_ephemeral})
        return call_messages

    def restore_history(self, history: list[dict[str, str]]) -> None:
        """Restore conversation_history from a previously saved session.

        Used by the resume path to reconstruct the SimulatedUser's memory
        of prior escalation exchanges. The provided history replaces the
        current conversation_history (which only contains the system message
        at construction time).

        The history list must include the system message as history[0] and
        alternating user/assistant turns from prior escalation exchanges.

        Args:
            history: Complete conversation history including system message.
        """
        self.conversation_history = list(history)

    # ── [FINISH] verification ──

    def _verify_finish(self) -> str:
        """Check whether all research goals are covered by workspace conclusions.

        Returns the LLM auditor's verdict: "COMPLETE" or "INCOMPLETE: ...".
        Returns "INCOMPLETE: no conclusions" when workspace has none.
        On LLM call failure, conservatively returns INCOMPLETE so the
        run continues rather than crashing.
        """
        conclusions_summary = self._read_conclusions_summary()
        if not conclusions_summary:
            return "INCOMPLETE: no stage conclusions found in workspace"
        try:
            return self._verify_goal_coverage(conclusions_summary)
        except Exception as exc:
            logger.warning("Goal verification LLM call failed: %s", exc)
            return f"INCOMPLETE: verification failed ({type(exc).__name__}), assuming incomplete"

    def _read_conclusions_summary(self) -> str:
        """Read all stage conclusions from workspace into a summary string.

        Returns empty string if no project state or no conclusions exist.
        """
        state_path = self.workspace_root / "project_state.json"
        if not state_path.is_file():
            return ""

        state = json.loads(state_path.read_text(encoding="utf-8"))
        sections: list[str] = []

        for stage_id, stage_data in state.get("stages", {}).items():
            if not stage_data.get("has_conclusion"):
                continue
            conclusion_path = (
                self.workspace_root / "stages" / stage_id / "conclusion.md"
            )
            if conclusion_path.is_file():
                content = conclusion_path.read_text(encoding="utf-8")
                # Truncate long conclusions to keep the verification prompt manageable
                if len(content) > 500:
                    content = content[:500] + "\n[...truncated]"
                sections.append(f"### {stage_id}\n{content}")

        return "\n\n".join(sections)

    def _verify_goal_coverage(self, conclusions_summary: str) -> str:
        """Ask the LLM whether the brief's goals are fully covered.

        Returns the LLM's verdict: starts with "COMPLETE" if all goals
        are addressed, or "INCOMPLETE: ..." listing what's missing.
        """
        verification_messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a research completeness auditor. You will receive "
                    "a research brief containing numbered Research Goals, and a "
                    "summary of completed stage conclusions. Your job is to "
                    "determine whether ALL goals from the brief have been "
                    "addressed by the completed stages.\n\n"
                    "Respond with exactly one of:\n"
                    "- COMPLETE (if all goals are covered)\n"
                    "- INCOMPLETE: <list the goal numbers and brief descriptions "
                    "that are NOT yet addressed>"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Research Brief\n{self.research_brief}\n\n"
                    f"## Completed Stage Conclusions\n{conclusions_summary}"
                ),
            },
        ]

        result = self.llm_client.chat(
            messages=verification_messages, model=self.model
        )
        return (result.content or "INCOMPLETE: verification call returned empty").strip()

    # ── Artifact context ──

    # Upper bound on any single artifact's text embedded into the user-agent
    # LLM call. Real CSV artifacts in Phase B runs hit 44 MB, which blew past
    # mimo's context window and caused it to return empty content. Cap at
    # 50 KB (~500-1000 lines of CSV / dense text); when exceeded, show head
    # + tail plus a size note so the reviewer still gets the shape and a
    # sample to reason about.
    _MAX_ARTIFACT_BYTES: int = 50_000
    _LARGE_HEAD_LINES: int = 40
    _LARGE_TAIL_LINES: int = 10

    def _build_artifact_context(self, context: ReviewContext | None) -> str | None:
        """Build per-call artifact context without persisting it across turns."""

        if context is None or not context.artifact_paths:
            return None

        artifact_sections: list[str] = [
            "Review the following workspace artifacts before replying."
        ]
        _BINARY_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz", ".bz2"}

        for artifact_path in context.artifact_paths:
            try:
                absolute_path = resolve_path_within_root(self.workspace_root, artifact_path)
            except WorkspaceError:
                artifact_sections.append(f"\nArtifact path: {artifact_path}")
                artifact_sections.append("(path escapes workspace root)")
                continue

            artifact_sections.append(f"\nArtifact path: {artifact_path}")
            if not absolute_path.is_file():
                artifact_sections.append("(file not found)")
                continue
            if absolute_path.suffix.lower() in _BINARY_EXTENSIONS:
                size_kb = absolute_path.stat().st_size / 1024
                artifact_sections.append(f"(binary file, {size_kb:.1f} KB)")
                continue
            try:
                artifact_sections.extend(self._render_text_artifact(absolute_path))
            except UnicodeDecodeError:
                size_kb = absolute_path.stat().st_size / 1024
                artifact_sections.append(f"(binary file, {size_kb:.1f} KB)")

        return "\n".join(artifact_sections)

    def _render_text_artifact(self, absolute_path: Path) -> list[str]:
        """Render a text artifact, truncating to head+tail when too large.

        Files at or below _MAX_ARTIFACT_BYTES are embedded in full. Larger
        files are truncated to the first _LARGE_HEAD_LINES and last
        _LARGE_TAIL_LINES with an elision marker and a size note, so the
        reviewer sees shape and a sample without overflowing the LLM's
        context window.
        """
        file_size = absolute_path.stat().st_size
        if file_size <= self._MAX_ARTIFACT_BYTES:
            return ["Artifact content:", absolute_path.read_text(encoding="utf-8")]

        text = absolute_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        total_lines = len(lines)
        head = lines[: self._LARGE_HEAD_LINES]
        tail = (
            lines[-self._LARGE_TAIL_LINES :]
            if self._LARGE_TAIL_LINES and total_lines > self._LARGE_HEAD_LINES
            else []
        )
        size_kb = file_size / 1024
        note = (
            f"(large file, {size_kb:.1f} KB / {total_lines} lines — showing "
            f"first {len(head)} and last {len(tail)} lines)"
        )
        body_parts: list[str] = ["Artifact content (truncated):", note]
        body_parts.extend(head)
        if tail:
            omitted = total_lines - len(head) - len(tail)
            body_parts.append(f"... [{omitted} lines omitted] ...")
            body_parts.extend(tail)
        return body_parts
