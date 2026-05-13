"""Tests for HumanUser and SimulatedUser implementations."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agent.schemas import LLMResponse
from research_agent.schemas import ReviewContext
from research_agent.user_agent import HumanUser
from research_agent.user_agent import SimulatedUser


class MockLLMClient:
    """Record chat calls and return predefined text responses."""

    def __init__(self, responses: list[str]) -> None:
        """Initialize the mock with a queue of response texts."""

        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> LLMResponse:
        """Record kwargs and return the next queued response."""

        self.calls.append(kwargs)
        return LLMResponse(content=self._responses.pop(0))


def _expected_simulated_user_system_prompt(research_brief: str) -> str:
    """Build the expected SimulatedUser system prompt for assertions."""

    return (
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
    )


def test_human_user_respond_prints_message_and_reads_input(capsys: Any, tmp_path: Path) -> None:
    """HumanUser should print the agent message and return terminal input."""

    user = HumanUser(workspace_root=tmp_path)

    with patch("builtins.input", return_value="I will review it.") as mock_input:
        response = user.respond("Please review the plan.")

    captured = capsys.readouterr().out
    assert response == "I will review it."
    assert "Agent: Please review the plan." in captured
    assert "─" * 60 in captured
    mock_input.assert_called_once_with("You: ")


def test_human_user_respond_shows_absolute_artifact_paths(capsys: Any, tmp_path: Path) -> None:
    """HumanUser should display workspace-resolved artifact paths when provided."""

    user = HumanUser(workspace_root=tmp_path)
    context = ReviewContext(artifact_paths=["plans/stage_01_plan.md", "stages/stage_01/report.md"])

    with patch("builtins.input", return_value="Looks good."):
        user.respond("Review these files.", context=context)

    captured = capsys.readouterr().out
    assert "Review artifacts:" in captured
    assert str(tmp_path / "plans/stage_01_plan.md") in captured
    assert str(tmp_path / "stages/stage_01/report.md") in captured


def test_simulated_user_respond_uses_research_brief_as_system_message(tmp_path: Path) -> None:
    """SimulatedUser should seed the conversation with a system message from the brief."""

    llm_client = MockLLMClient(responses=["First reply"])
    user = SimulatedUser(
        research_brief="Study CRISPR safety trade-offs.",
        llm_client=llm_client,  # type: ignore[arg-type]
        model="gpt-test",
        workspace_root=tmp_path,
    )

    response = user.respond("What should we focus on first?")

    assert response == "First reply"
    assert len(llm_client.calls) == 1
    call = llm_client.calls[0]
    assert call["model"] == "gpt-test"
    assert "tools" not in call
    messages = call["messages"]
    assert messages[0]["role"] == "system"
    assert "Study CRISPR safety trade-offs." in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "What should we focus on first?"}


def test_simulated_user_conversation_history_grows_across_turns(tmp_path: Path) -> None:
    """SimulatedUser should retain prior user/assistant turns across responds."""

    llm_client = MockLLMClient(responses=["First answer", "Second answer"])
    user = SimulatedUser(
        research_brief="Investigate catalyst benchmarks.",
        llm_client=llm_client,  # type: ignore[arg-type]
        model="gpt-test",
        workspace_root=tmp_path,
    )

    first_response = user.respond("Turn one question")
    second_response = user.respond("Turn two question")

    assert first_response == "First answer"
    assert second_response == "Second answer"
    assert user.conversation_history == [
        {
            "role": "system",
            "content": _expected_simulated_user_system_prompt(
                "Investigate catalyst benchmarks."
            ),
        },
        {"role": "user", "content": "Turn one question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Turn two question"},
        {"role": "assistant", "content": "Second answer"},
    ]
    second_call_messages = llm_client.calls[1]["messages"]
    assert second_call_messages[-1] == {"role": "user", "content": "Turn two question"}
    assert second_call_messages[-2] == {"role": "assistant", "content": "First answer"}


def test_simulated_user_injects_artifact_content_without_persisting_it(tmp_path: Path) -> None:
    """Artifact file contents should be injected for one call only, not stored in history."""

    artifact_path = tmp_path / "stages" / "stage_01" / "notes.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("# Notes\nImportant finding", encoding="utf-8")

    llm_client = MockLLMClient(responses=["Reviewed the artifact.", "No artifact this turn."])
    user = SimulatedUser(
        research_brief="Assess artifact handling.",
        llm_client=llm_client,  # type: ignore[arg-type]
        model="gpt-test",
        workspace_root=tmp_path,
    )
    context = ReviewContext(artifact_paths=["stages/stage_01/notes.md"])

    user.respond("Please inspect the artifact.", context=context)
    user.respond("What do you remember now?")

    first_call_messages = llm_client.calls[0]["messages"]
    assert len(first_call_messages) == 3
    ephemeral_message = first_call_messages[2]
    assert ephemeral_message["role"] == "user"
    assert "Artifact path: stages/stage_01/notes.md" in ephemeral_message["content"]
    assert "# Notes\nImportant finding" in ephemeral_message["content"]

    assert user.conversation_history == [
        {
            "role": "system",
            "content": _expected_simulated_user_system_prompt("Assess artifact handling."),
        },
        {"role": "user", "content": "Please inspect the artifact."},
        {"role": "assistant", "content": "Reviewed the artifact."},
        {"role": "user", "content": "What do you remember now?"},
        {"role": "assistant", "content": "No artifact this turn."},
    ]

    second_call_messages = llm_client.calls[1]["messages"]
    assert len(second_call_messages) == 4
    assert all("# Notes\nImportant finding" not in message["content"] for message in second_call_messages)


def test_simulated_user_marks_artifact_paths_outside_workspace(tmp_path: Path) -> None:
    """Escaped artifact paths should not be read from outside the workspace root."""

    outside_path = tmp_path.parent / "outside_notes.md"
    outside_path.write_text("do not leak", encoding="utf-8")

    llm_client = MockLLMClient(responses=["Acknowledged."])
    user = SimulatedUser(
        research_brief="Assess path safety.",
        llm_client=llm_client,  # type: ignore[arg-type]
        model="gpt-test",
        workspace_root=tmp_path,
    )

    user.respond("Review this file.", context=ReviewContext(artifact_paths=["../outside_notes.md"]))

    artifact_message = llm_client.calls[0]["messages"][-1]["content"]
    assert "(path escapes workspace root)" in artifact_message
    assert "do not leak" not in artifact_message


def test_simulated_user_truncates_large_artifacts(tmp_path: Path) -> None:
    """Artifacts over _MAX_ARTIFACT_BYTES should be truncated head+tail.

    Regression for 2026-04-18 bug: Phase B scenarios crashed when the main
    agent passed multi-MB CSVs as artifact_paths — SimulatedUser read the
    full file into mimo's context, blowing past its window and causing
    mimo to return empty content for 3+ retries in a row. The fix caps
    artifact text at _MAX_ARTIFACT_BYTES with head/tail truncation.
    """

    large_lines = [f"row_{i},val_{i}" for i in range(5000)]
    large_csv = tmp_path / "stages" / "stage1" / "big.csv"
    large_csv.parent.mkdir(parents=True, exist_ok=True)
    large_csv.write_text("\n".join(large_lines), encoding="utf-8")

    assert large_csv.stat().st_size > SimulatedUser._MAX_ARTIFACT_BYTES

    llm_client = MockLLMClient(responses=["Reviewed."])
    user = SimulatedUser(
        research_brief="Review large artifact.",
        llm_client=llm_client,  # type: ignore[arg-type]
        model="gpt-test",
        workspace_root=tmp_path,
    )

    user.respond(
        "Check this CSV.",
        context=ReviewContext(artifact_paths=["stages/stage1/big.csv"]),
    )

    ephemeral = llm_client.calls[0]["messages"][-1]["content"]
    assert "Artifact content (truncated):" in ephemeral
    assert "large file" in ephemeral
    assert "lines omitted" in ephemeral
    # Head lines should be present, middle rows should not.
    assert "row_0,val_0" in ephemeral
    assert "row_2500,val_2500" not in ephemeral
    # The ephemeral content must be far smaller than the original file.
    assert len(ephemeral) < SimulatedUser._MAX_ARTIFACT_BYTES + 5000


def test_simulated_user_keeps_small_artifacts_intact(tmp_path: Path) -> None:
    """Small artifacts must not be truncated — only oversize ones get head/tail."""

    small_md = tmp_path / "stages" / "stage1" / "conclusion.md"
    small_md.parent.mkdir(parents=True, exist_ok=True)
    body = "# Conclusion\n" + "Finding lines.\n" * 20
    small_md.write_text(body, encoding="utf-8")

    assert small_md.stat().st_size < SimulatedUser._MAX_ARTIFACT_BYTES

    llm_client = MockLLMClient(responses=["Reviewed."])
    user = SimulatedUser(
        research_brief="Review small artifact.",
        llm_client=llm_client,  # type: ignore[arg-type]
        model="gpt-test",
        workspace_root=tmp_path,
    )

    user.respond(
        "Check this markdown.",
        context=ReviewContext(artifact_paths=["stages/stage1/conclusion.md"]),
    )

    ephemeral = llm_client.calls[0]["messages"][-1]["content"]
    assert "Artifact content:" in ephemeral
    assert "truncated" not in ephemeral
    assert body in ephemeral


def test_simulated_user_respond_without_context_works(tmp_path: Path) -> None:
    """SimulatedUser should handle calls without review context."""

    llm_client = MockLLMClient(responses=["Plain reply"])
    user = SimulatedUser(
        research_brief="General-purpose brief.",
        llm_client=llm_client,  # type: ignore[arg-type]
        model="gpt-test",
        workspace_root=tmp_path,
    )

    response = user.respond("Just answer plainly.", context=None)

    assert response == "Plain reply"
    call = llm_client.calls[0]
    assert call["messages"] == [
        {
            "role": "system",
            "content": _expected_simulated_user_system_prompt("General-purpose brief."),
        },
        {"role": "user", "content": "Just answer plainly."},
    ]
