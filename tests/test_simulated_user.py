"""Tests for SimulatedUser [FINISH] gate mechanism."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agent.schemas import LLMResponse
from research_agent.user_agent.simulated import SimulatedUser


# ── Helpers ──


def _mock_llm_client(*responses: str) -> MagicMock:
    """Create a mock LLM client that returns responses in order."""
    client = MagicMock()
    client.chat.side_effect = [LLMResponse(content=r) for r in responses]
    return client


def _setup_workspace_with_conclusions(
    ws: Path, stage_conclusions: dict[str, str]
) -> None:
    """Create a workspace with project_state.json and conclusion files."""
    stages = {}
    for stage_id, conclusion_text in stage_conclusions.items():
        stages[stage_id] = {
            "stage_id": stage_id,
            "has_conclusion": True,
            "plan_status": "approved",
            "plan_reviewed": True,
            "task_ids": ["t1"],
        }
        stage_dir = ws / "stages" / stage_id
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "conclusion.md").write_text(conclusion_text)

    (ws / "project_state.json").write_text(json.dumps({"stages": stages}))


# ── _read_conclusions_summary tests ──


def test_read_conclusions_returns_empty_when_no_state() -> None:
    """No project_state.json → empty string."""
    user = SimulatedUser("brief", MagicMock(), "test", Path("/nonexistent"))
    assert user._read_conclusions_summary() == ""


def test_read_conclusions_returns_stage_summaries(tmp_path: Path) -> None:
    """Conclusions from workspace are assembled into a summary."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    _setup_workspace_with_conclusions(ws, {
        "stage_01": "# OD workflow done\nDetails...",
        "stage_02": "# Growth parameterization done",
    })

    user = SimulatedUser("brief", MagicMock(), "test", ws)
    summary = user._read_conclusions_summary()

    assert "stage_01" in summary
    assert "OD workflow done" in summary
    assert "stage_02" in summary


# ── [FINISH] gate tests ──


def test_finish_allowed_when_verification_says_complete(tmp_path: Path) -> None:
    """[FINISH] passes through when LLM auditor returns COMPLETE."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    _setup_workspace_with_conclusions(ws, {"s1": "# Done"})

    # respond call → auditor verification call
    client = _mock_llm_client("All done! [FINISH]", "COMPLETE")
    user = SimulatedUser("## Research Goals\n1. A\n", client, "test", ws)

    result = user.respond("Final results.")

    assert "[FINISH]" in result
    assert client.chat.call_count == 2


def test_finish_blocked_triggers_recall(tmp_path: Path) -> None:
    """When incomplete, LLM is re-called and agent gets a natural response."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    _setup_workspace_with_conclusions(ws, {"s1": "# Partial work"})

    # respond → auditor (incomplete) → re-call with nudge
    client = _mock_llm_client(
        "Looks good! [FINISH]",
        "INCOMPLETE: Goal 2 and Goal 3 not addressed",
        "We still need Goal 2 and Goal 3. Let's start with Goal 2.",
    )
    user = SimulatedUser("## Research Goals\n1. A\n2. B\n3. C\n", client, "test", ws)

    result = user.respond("Here are results.")

    assert "[FINISH]" not in result
    assert "Goal 2" in result
    # Agent sees natural language, no system nudges
    assert "[System:" not in result
    assert "blocked" not in result.lower()
    assert client.chat.call_count == 3


def test_finish_blocked_retries_when_recall_still_has_finish(tmp_path: Path) -> None:
    """If re-called LLM still emits [FINISH], retry until a valid response."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    _setup_workspace_with_conclusions(ws, {"s1": "# Partial"})

    client = _mock_llm_client(
        "Done! [FINISH]",
        "INCOMPLETE: Goal 2 missing",
        "Let's do Goal 2. [FINISH]",      # retry 1 still bad
        "More work to do. [FINISH]",      # retry 2 still bad
        "Let's tackle Goal 2 next.",      # retry 3 valid
    )
    user = SimulatedUser("## Research Goals\n1. A\n2. B\n", client, "test", ws)

    result = user.respond("Results.")

    assert "[FINISH]" not in result
    assert "Goal 2" in result
    assert client.chat.call_count == 5


def test_finish_blocked_raises_after_max_retries(tmp_path: Path) -> None:
    """Raise instead of sending empty/[FINISH] content to the main agent."""
    import pytest

    ws = tmp_path / "workspace"
    ws.mkdir()
    _setup_workspace_with_conclusions(ws, {"s1": "# Partial"})

    client = _mock_llm_client(
        "Done! [FINISH]",
        "INCOMPLETE: Goal 2 missing",
        "[FINISH]",  # retry 1 stripped → empty
        "",          # retry 2 empty
        "[FINISH]",  # retry 3 still bad
    )
    user = SimulatedUser("## Research Goals\n1. A\n2. B\n", client, "test", ws)

    with pytest.raises(RuntimeError, match="non-\\[FINISH\\]"):
        user.respond("Results.")


def test_finish_blocked_raises_on_empty_recall(tmp_path: Path) -> None:
    """Empty re-call response must not leak to main agent."""
    import pytest

    ws = tmp_path / "workspace"
    ws.mkdir()
    _setup_workspace_with_conclusions(ws, {"s1": "# Partial"})

    client = _mock_llm_client(
        "Done! [FINISH]",
        "INCOMPLETE: Goal 2 missing",
        "",  # retry 1 empty
        "",  # retry 2 empty
        "",  # retry 3 empty
    )
    user = SimulatedUser("## Research Goals\n1. A\n2. B\n", client, "test", ws)

    with pytest.raises(RuntimeError):
        user.respond("Results.")


def test_finish_blocked_when_no_conclusions(tmp_path: Path) -> None:
    """[FINISH] blocked and re-called when no conclusions exist."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "project_state.json").write_text(json.dumps({"stages": {}}))

    # respond → no conclusions so no auditor call → re-call with nudge
    client = _mock_llm_client(
        "Nothing to do! [FINISH]",
        "We haven't started any work yet. Let's begin with Goal 1.",
    )
    user = SimulatedUser("## Research Goals\n1. A\n", client, "test", ws)

    result = user.respond("Anything?")

    assert "[FINISH]" not in result
    assert "Goal 1" in result
    # No auditor call when zero conclusions — respond + re-call = 2
    assert client.chat.call_count == 2


def test_no_gate_when_no_finish_marker() -> None:
    """Normal responses without [FINISH] pass through without verification."""
    client = _mock_llm_client("Approved, go ahead.")
    user = SimulatedUser("brief", client, "test", Path("/fake"))

    result = user.respond("Plan for review.")

    assert result == "Approved, go ahead."
    assert client.chat.call_count == 1


# ── Empty-response retry tests ──


def test_empty_response_retries_until_valid() -> None:
    """Empty LLM content triggers a retry instead of leaking to the main agent."""
    client = _mock_llm_client("", "", "Here's what to do next.")
    user = SimulatedUser("brief", client, "test", Path("/fake"))

    result = user.respond("What's next?")

    assert result == "Here's what to do next."
    assert client.chat.call_count == 3


def test_empty_response_raises_after_max_retries() -> None:
    """All retries empty → raise RuntimeError; main agent never sees empty."""
    import pytest

    client = _mock_llm_client("", "", "")
    user = SimulatedUser("brief", client, "test", Path("/fake"))

    with pytest.raises(RuntimeError, match="empty content"):
        user.respond("What's next?")

    assert client.chat.call_count == 3


def test_empty_response_retry_nudge_injected(tmp_path: Path) -> None:
    """On retry, an ephemeral [Retry N] nudge is appended to the call messages.

    Verifies that retries don't just re-send the same context — they supply
    a visible nudge so the LLM understands why it's being re-called.
    """
    client = _mock_llm_client("", "Finally a real answer.")
    user = SimulatedUser("brief", client, "test", Path("/fake"))

    user.respond("Anything?")

    # Second call (the retry) should carry the nudge as a trailing user msg.
    second_call = client.chat.call_args_list[1]
    messages = second_call.kwargs.get("messages") or second_call.args[0]
    last_msg = messages[-1]
    assert last_msg["role"] == "user"
    assert "[Retry 1]" in last_msg["content"]
    assert "empty" in last_msg["content"]


# ── respond() integration tests ──


def test_respond_preserves_conversation_history() -> None:
    """conversation_history grows correctly across turns."""
    client = _mock_llm_client("First reply.", "Second reply.")
    user = SimulatedUser("brief", client, "test", Path("/fake"))

    user.respond("Hello.")
    user.respond("Next.")

    assert len(user.conversation_history) == 5
    assert user.conversation_history[1] == {"role": "user", "content": "Hello."}
    assert user.conversation_history[2] == {"role": "assistant", "content": "First reply."}


def test_blocked_finish_stores_recall_in_history(tmp_path: Path) -> None:
    """When [FINISH] is blocked, the re-called response is what goes into history."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    _setup_workspace_with_conclusions(ws, {"s1": "# Partial"})

    client = _mock_llm_client(
        "All done! [FINISH]",
        "INCOMPLETE: Goal 2 missing",
        "Let's do Goal 2 next.",
    )
    user = SimulatedUser("## Research Goals\n1. A\n2. B\n", client, "test", ws)

    user.respond("Results.")

    # History should have the re-called response, not the [FINISH] one
    last = user.conversation_history[-1]
    assert last["role"] == "assistant"
    assert "Goal 2" in last["content"]
    assert "[FINISH]" not in last["content"]


def test_system_prompt_has_no_goal_tracking() -> None:
    """System prompt stays generic — no goal progress injection."""
    client = _mock_llm_client("Ok.")
    user = SimulatedUser("## Research Goals\n1. A\n2. B\n", client, "test", Path("/fake"))

    user.respond("Do something.")

    call_kwargs = client.chat.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
    system_content = messages[0]["content"]
    assert "Progress" not in system_content
    assert "[x]" not in system_content
    assert "pending" not in system_content.lower()


def test_system_prompt_teaches_finish_emission() -> None:
    """System prompt must instruct the LLM when and how to emit [FINISH].

    Regression for 2026-04-18 bug: kaggle run hit 58 escalations because the
    simulated user kept saying 'no further action needed' without ever emitting
    `[FINISH]`. The _verify_finish gate only validates [FINISH] after it is
    emitted — nothing taught the LLM to emit it in the first place.
    """

    client = _mock_llm_client("Ok.")
    user = SimulatedUser("## Research Goals\n1. A\n", client, "test", Path("/fake"))

    user.respond("Any further action?")

    call_kwargs = client.chat.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
    system_content = messages[0]["content"]

    # The prompt MUST mention [FINISH] explicitly so the LLM knows the token.
    assert "[FINISH]" in system_content
    # The prompt MUST clarify that natural-language confirmation is not enough.
    natural_hints = ("no further action", "publication-ready", "project is complete")
    assert any(hint in system_content for hint in natural_hints), (
        "prompt should warn that natural-language confirmations do not end the session"
    )
    # The prompt should reference numbered goals to nudge the model to check coverage.
    assert "Research Goals" in system_content
