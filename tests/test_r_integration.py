"""Integration test — R language support end-to-end.

Validates the full R path: probe → tool registration → R code execution
→ file output detection → harness collection.

Uses a ScriptedLLM with real workspace, real R probe, real Rscript execution.

Flow (9 LLM calls):
1. Main: save_plan("stage_01", ...)         → survival analysis plan
2. Main: escalate_to_user(...)              → user approval
3. Main: approve_plan("stage_01")           → plan approved
4. Main: dispatch_subagent(task_001, ...)   → R survival analysis task
5. Sub:  run_code(language="r", ...)        → Kaplan-Meier + Cox regression
6. Sub:  finish_task(...)                   → sub agent completes
7. Main: save_conclusion("stage_01", ...)   → conclusion
8. Main: escalate_to_user(...)              → present findings
9. Main: finish_run(...)                    → main loop completes
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agent.harness import ResearchHarness
from research_agent.llm_client import LLMClient
from research_agent.runtime_env import RProbeResult, RuntimeProbeResult
from research_agent.schemas import LLMResponse, ToolCallRequest
from research_agent.workspace import Workspace

pytestmark = pytest.mark.skipif(
    not shutil.which("Rscript"),
    reason="Rscript not available",
)


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

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_index = 0

    def respond(self, message: str, context: object | None = None) -> str:
        del message, context
        response = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1
        return response


def _tc(call_id: str, name: str, **kwargs: Any) -> ToolCallRequest:
    return ToolCallRequest(id=call_id, name=name, arguments=kwargs)


# ── R survival analysis code ──

R_SURVIVAL_CODE = """\
# Generate synthetic survival data
set.seed(42)
n <- 60
df <- data.frame(
  patient_id = 1:n,
  time = rexp(n, rate=0.05),
  status = rbinom(n, 1, 0.6),
  treatment = rep(c("control", "drug"), each=n/2),
  age = rnorm(n, mean=55, sd=10)
)

# Kaplan-Meier survival curves
library(survival)
km_fit <- survfit(Surv(time, status) ~ treatment, data=df)
cat("Kaplan-Meier summary:\\n")
print(summary(km_fit)$table)

# Log-rank test
lr_test <- survdiff(Surv(time, status) ~ treatment, data=df)
cat("\\nLog-rank p-value:", 1 - pchisq(lr_test$chisq, 1), "\\n")

# Cox proportional hazards
cox_fit <- coxph(Surv(time, status) ~ treatment + age, data=df)
cat("\\nCox regression:\\n")
print(summary(cox_fit)$coefficients)

# Save results
write.csv(df, "survival_data.csv", row.names=FALSE)
cox_summary <- data.frame(
  variable = rownames(summary(cox_fit)$coefficients),
  HR = exp(coef(cox_fit)),
  pvalue = summary(cox_fit)$coefficients[, "Pr(>|z|)"]
)
write.csv(cox_summary, "cox_results.csv", row.names=FALSE)
cat("\\nFiles written: survival_data.csv, cox_results.csv\\n")
"""


# ── Scripted responses ──


def _build_r_survival_responses() -> list[LLMResponse]:
    """9-response sequence: single-stage R survival analysis."""
    return [
        # 1. Main: save_plan
        LLMResponse(
            content="I'll create a plan for survival analysis using R.",
            tool_calls=[
                _tc(
                    "r_01",
                    "save_plan",
                    stage_id="stage_01",
                    content=(
                        "# Stage 1: Survival Analysis\n\n"
                        "## Objective\nKaplan-Meier + Cox regression on patient cohort.\n\n"
                        "## Method\nR survival package: survfit, survdiff, coxph.\n"
                    ),
                ),
            ],
        ),
        # 2. Main: escalate_to_user
        LLMResponse(
            content="Let me get your feedback.",
            tool_calls=[
                _tc(
                    "r_02",
                    "escalate_to_user",
                    summary="Plan: Kaplan-Meier + Cox regression using R survival package.",
                    stage_id="stage_01",
                    artifact_paths=["plans/stage_01_plan.md"],
                ),
            ],
        ),
        # 3. Main: approve_plan
        LLMResponse(
            content="Approving.",
            tool_calls=[_tc("r_03", "approve_plan", stage_id="stage_01")],
        ),
        # 4. Main: dispatch_subagent
        LLMResponse(
            content="Dispatching R survival analysis task.",
            tool_calls=[
                _tc(
                    "r_04",
                    "dispatch_subagent",
                    task_id="task_survival",
                    stage_id="stage_01",
                    task_description=(
                        "Run Kaplan-Meier survival analysis and Cox regression "
                        "on synthetic patient data using R. Save results as CSV."
                    ),
                    role="data_analyst",
                    max_turns=10,
                ),
            ],
        ),
        # 5. Sub: run_code in R
        LLMResponse(
            content=None,
            tool_calls=[
                _tc(
                    "r_05",
                    "run_code",
                    code=R_SURVIVAL_CODE,
                    language="r",
                ),
            ],
        ),
        # 6. Sub: finish_task
        LLMResponse(
            content="Analysis complete.",
            tool_calls=[
                _tc(
                    "r_06",
                    "finish_task",
                    summary=(
                        "Survival analysis complete. Kaplan-Meier curves and "
                        "Cox regression fitted. Results saved to CSV."
                    ),
                    blockers=[],
                    suggestions=["Consider adding forest plot visualization."],
                    artifact_descriptions={
                        "survival_data.csv": "Synthetic patient survival data (n=60)",
                        "cox_results.csv": "Cox regression hazard ratios and p-values",
                    },
                ),
            ],
        ),
        # 7. Main: save_conclusion
        LLMResponse(
            content="Writing conclusion.",
            tool_calls=[
                _tc(
                    "r_07",
                    "save_conclusion",
                    stage_id="stage_01",
                    content=(
                        "# Conclusion\n\n"
                        "Survival analysis completed with R. "
                        "Cox model fitted with treatment and age covariates."
                    ),
                ),
            ],
        ),
        # 8. Main: escalate findings
        LLMResponse(
            content="Presenting results.",
            tool_calls=[
                _tc(
                    "r_08",
                    "escalate_to_user",
                    summary="Survival analysis complete. Cox regression results available.",
                    stage_id="stage_01",
                    artifact_paths=[
                        "stages/stage_01/conclusion.md",
                        "stages/stage_01/outputs/cox_results.csv",
                    ],
                ),
            ],
        ),
        # 9. Main: finish_run
        LLMResponse(
            content="Done.",
            tool_calls=[
                _tc(
                    "r_09",
                    "finish_run",
                    final_summary="Survival analysis stage complete.",
                ),
            ],
        ),
    ]


# ── Test ──


def test_r_survival_analysis_end_to_end(
    tmp_path: Path,
    real_probe_result: RuntimeProbeResult,
    real_r_probe_result: RProbeResult | None,
) -> None:
    """End-to-end: main agent dispatches sub-agent that runs R survival analysis."""

    assert real_r_probe_result is not None, "Rscript found but probe failed"

    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()

    scripted_llm = ScriptedLLM(_build_r_survival_responses())
    user_agent = StubUserAgent([
        "Analyze patient survival outcomes.",      # initial goal
        "Looks good, proceed with R analysis.",    # plan approval
        "Good results. Stop here.",                # conclusion review
    ])

    harness = ResearchHarness(
        model="scripted-r-test",
        workspace=workspace,
        user_agent=user_agent,
        llm_client=scripted_llm,
        system_prompt="You are a research agent.",
        max_turns=20,
        probe_result=real_probe_result,
        r_probe_result=real_r_probe_result,
    )

    result = harness.run()

    # ── RunResult ──
    assert result.stop_reason == "completed"
    assert result.stages_completed == 1
    assert result.escalation_count == 2

    # ── Workspace state ──
    state = workspace.get_state()
    assert "stage_01" in state.stages
    assert state.stages["stage_01"].stage_status == "completed"

    # ── R-generated output files exist ──
    outputs = workspace.root / "stages" / "stage_01" / "outputs"
    assert (outputs / "survival_data.csv").is_file()
    assert (outputs / "cox_results.csv").is_file()

    # ── Validate R output content ──
    import pandas as pd

    survival_df = pd.read_csv(outputs / "survival_data.csv")
    assert len(survival_df) == 60
    assert set(survival_df.columns) == {"patient_id", "time", "status", "treatment", "age"}
    assert set(survival_df["treatment"].unique()) == {"control", "drug"}

    cox_df = pd.read_csv(outputs / "cox_results.csv")
    assert "HR" in cox_df.columns
    assert "pvalue" in cox_df.columns
    assert len(cox_df) == 2  # treatment + age

    # ── Task result recorded correctly ──
    task_result_path = (
        workspace.root / "stages" / "stage_01" / "tasks" / "task_survival_result.json"
    )
    assert task_result_path.is_file()
    task_data = json.loads(task_result_path.read_text(encoding="utf-8"))
    assert task_data["status"] == "success"
    assert "survival" in task_data["summary"].lower()
    assert any("cox_results.csv" in p for p in task_data["artifact_paths"])

    # ── Artifact descriptions propagated ──
    assert task_data["artifact_descriptions"]["cox_results.csv"] == (
        "Cox regression hazard ratios and p-values"
    )

    # ── Session log ──
    session_data = json.loads(
        (workspace.root / "session.json").read_text(encoding="utf-8")
    )
    assert len(session_data["subagent_runs"]) == 1
    assert session_data["subagent_runs"][0]["task_id"] == "task_survival"


def test_r_and_python_mixed_execution(
    tmp_path: Path,
    real_probe_result: RuntimeProbeResult,
    real_r_probe_result: RProbeResult | None,
) -> None:
    """Sub-agent uses both Python and R in the same task — data exchanged via CSV."""

    assert real_r_probe_result is not None

    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()

    responses = [
        # 1. Main: save_plan
        LLMResponse(
            content="Planning mixed analysis.",
            tool_calls=[
                _tc("m_01", "save_plan", stage_id="stage_01",
                     content="# Mixed Python+R analysis"),
            ],
        ),
        # 2. Main: escalate
        LLMResponse(
            content="Review.", tool_calls=[
                _tc("m_02", "escalate_to_user", summary="Plan ready.",
                     stage_id="stage_01"),
            ],
        ),
        # 3. Main: approve
        LLMResponse(
            content="Approved.", tool_calls=[
                _tc("m_03", "approve_plan", stage_id="stage_01"),
            ],
        ),
        # 4. Main: dispatch
        LLMResponse(
            content="Dispatch.", tool_calls=[
                _tc("m_04", "dispatch_subagent", task_id="task_mixed",
                     stage_id="stage_01",
                     task_description="Generate data in Python, analyze in R.",
                     max_turns=10),
            ],
        ),
        # 5. Sub: Python generates data
        LLMResponse(
            content=None, tool_calls=[
                _tc("m_05", "run_code", language="python",
                     code=(
                         "import pandas as pd\n"
                         "import numpy as np\n"
                         "np.random.seed(42)\n"
                         "df = pd.DataFrame({'x': np.random.randn(50), 'y': np.random.randn(50)})\n"
                         "df.to_csv('input_data.csv', index=False)\n"
                         "print(f'Generated {len(df)} rows')"
                     )),
            ],
        ),
        # 6. Sub: R reads CSV and computes correlation
        LLMResponse(
            content=None, tool_calls=[
                _tc("m_06", "run_code", language="r",
                     code=(
                         'df <- read.csv("input_data.csv")\n'
                         'cat("Rows:", nrow(df), "\\n")\n'
                         'cor_result <- cor.test(df$x, df$y)\n'
                         'cat("Correlation:", cor_result$estimate, "\\n")\n'
                         'cat("P-value:", cor_result$p.value, "\\n")\n'
                         'result <- data.frame(\n'
                         '  statistic=c("correlation", "p_value"),\n'
                         '  value=c(cor_result$estimate, cor_result$p.value)\n'
                         ')\n'
                         'write.csv(result, "r_analysis.csv", row.names=FALSE)\n'
                     )),
            ],
        ),
        # 7. Sub: finish_task
        LLMResponse(
            content="Done.", tool_calls=[
                _tc("m_07", "finish_task",
                     summary="Python generated data, R computed correlation.",
                     blockers=[], suggestions=[]),
            ],
        ),
        # 8. Main: save_conclusion
        LLMResponse(
            content="Concluding.", tool_calls=[
                _tc("m_08", "save_conclusion", stage_id="stage_01",
                     content="# Done\nMixed analysis complete."),
            ],
        ),
        # 9. Main: escalate
        LLMResponse(
            content="Results.", tool_calls=[
                _tc("m_09", "escalate_to_user",
                     summary="Mixed analysis done.", stage_id="stage_01"),
            ],
        ),
        # 10. Main: finish
        LLMResponse(
            content="Finished.", tool_calls=[
                _tc("m_10", "finish_run", final_summary="All done."),
            ],
        ),
    ]

    scripted_llm = ScriptedLLM(responses)
    user_agent = StubUserAgent([
        "Run mixed analysis.",       # initial goal
        "Proceed.",                  # plan review
        "Stop.",                     # conclusion review
    ])

    harness = ResearchHarness(
        model="scripted-mixed",
        workspace=workspace,
        user_agent=user_agent,
        llm_client=scripted_llm,
        system_prompt="Test agent.",
        max_turns=20,
        probe_result=real_probe_result,
        r_probe_result=real_r_probe_result,
    )

    result = harness.run()

    assert result.stop_reason == "completed"

    outputs = workspace.root / "stages" / "stage_01" / "outputs"

    # Python-generated file exists
    assert (outputs / "input_data.csv").is_file()
    # R-generated file exists (R read Python's CSV and produced its own)
    assert (outputs / "r_analysis.csv").is_file()

    import pandas as pd
    r_result = pd.read_csv(outputs / "r_analysis.csv")
    assert set(r_result["statistic"]) == {"correlation", "p_value"}
    assert len(r_result) == 2
