"""Integration test — mock LLM, real everything else.

Uses a ScriptedLLM that returns pre-defined LLMResponse objects in sequence.
Everything else (workspace, tools, loop, harness) is real.

Flow (9 LLM calls):
1. Main: save_plan("stage_01", ...)         → creates stage, writes plan
2. Main: escalate_to_user(...)              → gets user approval
3. Main: approve_plan("stage_01")           → plan approved, stage executing
4. Main: dispatch_subagent(task_001, ...)   → enters sub loop
5. Sub:  run_code(...) + write_file(...)    → sub agent executes
6. Sub:  (no tool calls — final text)       → sub agent completes
7. Main: save_conclusion("stage_01", ...)   → writes conclusion
8. Main: escalate_to_user(...)              → present completed stage findings
9. Main: finish_run(...)                    → main loop completes
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agent.harness import ResearchHarness
from research_agent.llm_client import LLMClient
from research_agent.runtime_env import RuntimeProbeResult
from research_agent.schemas import LLMResponse, ToolCallRequest
from research_agent.workspace import Workspace


# ── Helpers ──


class ScriptedLLM(LLMClient):
    """LLMClient that returns pre-defined responses in order."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(default_model="scripted")
        self._responses = list(responses)
        self._call_index = 0

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        assert self._call_index < len(self._responses), (
            f"ScriptedLLM exhausted: called {self._call_index + 1} times "
            f"but only {len(self._responses)} responses scripted"
        )
        response = self._responses[self._call_index]
        self._call_index += 1
        return response


class StubUserAgent:
    """Returns a fixed response for escalation."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or ["Looks good, proceed."])
        self._call_index = 0

    def respond(self, message: str, context: object | None = None) -> str:
        del message, context
        response = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1
        return response


def _tc(call_id: str, name: str, **kwargs: Any) -> ToolCallRequest:
    """Shorthand to build a ToolCallRequest."""
    return ToolCallRequest(id=call_id, name=name, arguments=kwargs)


# ── Scripted responses ──


def _build_single_stage_responses() -> list[LLMResponse]:
    """Build the 9-response sequence for a single-stage integration test."""
    return [
        # 1. Main: save_plan
        LLMResponse(
            content="I'll create a plan for differential expression analysis.",
            tool_calls=[
                _tc(
                    "call_01",
                    "save_plan",
                    stage_id="stage_01",
                    content="# DE Analysis Plan\n\nRun DESeq2 on T cell counts.",
                ),
            ],
        ),
        # 2. Main: escalate_to_user
        LLMResponse(
            content="Let me get your feedback on this plan.",
            tool_calls=[
                _tc(
                    "call_02",
                    "escalate_to_user",
                    summary="Please review the DE analysis plan for stage_01.",
                    stage_id="stage_01",
                    artifact_paths=["plans/stage_01_plan.md"],
                ),
            ],
        ),
        # 3. Main: approve_plan
        LLMResponse(
            content="User approved. Approving the plan.",
            tool_calls=[
                _tc("call_03", "approve_plan", stage_id="stage_01"),
            ],
        ),
        # 4. Main: dispatch_subagent
        LLMResponse(
            content="Dispatching a data analyst sub agent.",
            tool_calls=[
                _tc(
                    "call_04",
                    "dispatch_subagent",
                    task_id="task_001",
                    stage_id="stage_01",
                    task_description="Run DE analysis on T cell data. Print DE gene count.",
                    role="data_analyst",
                    max_turns=10,
                ),
            ],
        ),
        # 5. Sub: run_code + write_file
        LLMResponse(
            content=None,
            tool_calls=[
                _tc(
                    "call_05a",
                    "run_code",
                    code="print('DE genes: 47')",
                    language="python",
                ),
                _tc(
                    "call_05b",
                    "write_file",
                    filename="results.csv",
                    content="gene,pval\nTP53,0.001\nBRCA1,0.003\n",
                ),
            ],
        ),
        # 6. Sub: terminal completion tool
        LLMResponse(
            content="Submitting the task result.",
            tool_calls=[
                _tc(
                    "call_06",
                    "finish_task",
                    summary="Analysis complete. Found 47 DE genes with FDR < 0.05.",
                    blockers=[],
                    suggestions=[],
                ),
            ],
        ),
        # 7. Main: save_conclusion
        LLMResponse(
            content="Writing the stage conclusion.",
            tool_calls=[
                _tc(
                    "call_07",
                    "save_conclusion",
                    stage_id="stage_01",
                    content="# Conclusion\n\n47 DE genes identified in T cell data.",
                ),
            ],
        ),
        # 8. Main: escalate completed-stage findings to the user
        LLMResponse(
            content="Presenting the completed stage findings to the user.",
            tool_calls=[
                _tc(
                    "call_08",
                    "escalate_to_user",
                    summary="Stage 01 is complete. Key result: 47 DE genes identified in T cell data.",
                    stage_id="stage_01",
                    artifact_paths=[
                        "stages/stage_01/conclusion.md",
                        "stages/stage_01/outputs/results.csv",
                    ],
                ),
            ],
        ),
        # 9. Main: terminal tool → main loop completes
        LLMResponse(
            content="The workflow is complete.",
            tool_calls=[
                _tc(
                    "call_09",
                    "finish_run",
                    final_summary="Stage 01 complete. Waiting for the user's next direction.",
                ),
            ],
        ),
    ]


# ── Test ──


def test_full_single_stage_flow(
    tmp_path: Path,
    real_probe_result: RuntimeProbeResult,
) -> None:
    """End-to-end single-stage flow with scripted LLM and real components."""
    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()

    scripted_llm = ScriptedLLM(_build_single_stage_responses())
    user_agent = StubUserAgent(
        [
            "Investigate T cell differential expression.",  # initial goal (generated by user_agent)
            "Looks good, proceed.",  # stage plan approval
            "Results look good. Stop after this stage.",  # stage conclusion review
        ]
    )

    harness = ResearchHarness(
        model="scripted-test",
        workspace=workspace,
        user_agent=user_agent,
        llm_client=scripted_llm,
        system_prompt="You are a test agent.",
        max_turns=20,
        probe_result=real_probe_result,
    )

    result = harness.run()

    # ── RunResult assertions ──
    assert result.stop_reason == "completed"
    assert result.stages_completed == 1
    assert result.escalation_count == 2
    assert result.tool_call_count > 0

    # ── Workspace file assertions ──
    state = workspace.get_state()
    assert "stage_01" in state.stages
    assert state.stages["stage_01"].stage_status == "completed"
    assert state.current_stage_id == "stage_01"

    assert (workspace.root / "plans" / "stage_01_plan.md").is_file()
    assert (workspace.root / "stages" / "stage_01" / "conclusion.md").is_file()
    assert (workspace.root / "stages" / "stage_01" / "outputs" / "results.csv").is_file()
    assert (
        workspace.root / "stages" / "stage_01" / "tasks" / "task_001_result.json"
    ).is_file()

    # Verify task result content
    task_result_path = (
        workspace.root / "stages" / "stage_01" / "tasks" / "task_001_result.json"
    )
    task_result_data = json.loads(task_result_path.read_text(encoding="utf-8"))
    assert task_result_data["task_id"] == "task_001"
    assert task_result_data["status"] == "success"
    assert "47 DE genes" in task_result_data["summary"]

    # ── Session log assertions ──
    session_path = workspace.root / "session.json"
    assert session_path.is_file()
    session_data = json.loads(session_path.read_text(encoding="utf-8"))
    assert session_data["model"] == "scripted-test"
    assert session_data["initial_goal"] == "Investigate T cell differential expression."
    assert session_data["run_result"]["stop_reason"] == "completed"
    assert session_data["project_state"]["stages"]["stage_01"]["stage_status"] == "completed"
    assert len(session_data["main_conversation"]) > 0
    assert len(session_data["subagent_runs"]) == 1
    assert session_data["subagent_runs"][0]["task_id"] == "task_001"
    assert session_data["subagent_runs"][0]["role"] == "data_analyst"
