"""Scenario test — 3-stage research flow with domain-specific user feedback.

Simulates a realistic bioinformatics workflow:
- Stage 1: T cell differential expression (user requests stricter FDR, flags candidate genes)
- Stage 2: Pathway enrichment (user redirects method and database choices)
- Stage 3: Cross-cell comparison (user specifies visualization preferences)

Mock data: tests/fixtures/ contains expression matrices and metadata that
subagents actually read via read_file during execution.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agent.harness import ResearchHarness
from research_agent.llm_client import LLMClient
from research_agent.runtime_env import RuntimeProbeResult
from research_agent.schemas import LLMResponse, ToolCallRequest
from research_agent.workspace import Workspace

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


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
            f"ScriptedLLM exhausted at call {self._call_index + 1} "
            f"(only {len(self._responses)} scripted)"
        )
        response = self._responses[self._call_index]
        self._call_index += 1
        return response


class ScriptedUserAgent:
    """Returns scripted natural-language responses for escalation calls."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_index = 0

    def respond(self, message: str, context: object | None = None) -> str:
        assert self._call_index < len(self._responses), (
            f"ScriptedUserAgent exhausted at call {self._call_index + 1}"
        )
        response = self._responses[self._call_index]
        self._call_index += 1
        return response


def _tc(call_id: str, name: str, **kwargs: Any) -> ToolCallRequest:
    """Shorthand to build a ToolCallRequest."""
    return ToolCallRequest(id=call_id, name=name, arguments=kwargs)


def _setup_workspace(tmp_path: Path) -> Workspace:
    """Create workspace and populate with mock data from fixtures."""
    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()

    # Copy data catalog
    shutil.copy(FIXTURES_DIR / "data_catalog.json", workspace.root / "data_catalog.json")

    # Copy expression data
    data_dir = workspace.root / "data"
    data_dir.mkdir()
    shutil.copy(FIXTURES_DIR / "tcell_counts.csv", data_dir / "tcell_counts.csv")
    shutil.copy(FIXTURES_DIR / "bcell_counts.csv", data_dir / "bcell_counts.csv")
    shutil.copy(FIXTURES_DIR / "sample_metadata.csv", data_dir / "sample_metadata.csv")

    return workspace


# ── Stage 1: T cell differential expression ──
# User feedback: stricter FDR, flag TP53/BRCA1, check immune genes
# Tasks: data_analyst (DE analysis) + visualization (volcano plot)


def _stage1_responses() -> list[LLMResponse]:
    return [
        # 1. save_plan
        LLMResponse(
            content="I'll start by analyzing differential expression in T cells.",
            tool_calls=[
                _tc(
                    "s1_01",
                    "save_plan",
                    stage_id="stage_01",
                    content=(
                        "# Stage 1: T Cell Differential Expression\n\n"
                        "## Objective\n"
                        "Identify differentially expressed genes between treatment and control "
                        "in T cell samples using DESeq2.\n\n"
                        "## Data\n"
                        "- Input: data/tcell_counts.csv (10 genes x 6 samples)\n"
                        "- Metadata: data/sample_metadata.csv\n\n"
                        "## Method\n"
                        "- DESeq2 with FDR < 0.05 correction\n"
                        "- Volcano plot of results\n"
                    ),
                ),
            ],
        ),
        # 2. escalate — ask user to review plan
        LLMResponse(
            content="Let me get your input on the analysis plan.",
            tool_calls=[
                _tc(
                    "s1_02",
                    "escalate_to_user",
                    summary=(
                        "I've drafted the DE analysis plan for T cells. "
                        "I see 6 samples (3 control, 3 treatment) with 10 genes. "
                        "Plan: DESeq2 with FDR < 0.05. Should I proceed with these settings?"
                    ),
                    stage_id="stage_01",
                    artifact_paths=["plans/stage_01_plan.md"],
                ),
            ],
        ),
        # 3. approve_plan (after user gives stricter FDR guidance)
        LLMResponse(
            content=(
                "User requested FDR < 0.01 and wants TP53/BRCA1 flagged. "
                "Updating and approving the plan."
            ),
            tool_calls=[_tc("s1_03", "approve_plan", stage_id="stage_01")],
        ),
        # 4. dispatch DE analysis task
        LLMResponse(
            content="Dispatching differential expression analysis with user's parameters.",
            tool_calls=[
                _tc(
                    "s1_04",
                    "dispatch_subagent",
                    task_id="task_001",
                    stage_id="stage_01",
                    task_description=(
                        "Run differential expression analysis on T cell data.\n"
                        "1. Read data/tcell_counts.csv and data/sample_metadata.csv\n"
                        "2. Run DESeq2-style analysis (use FDR < 0.01 per user request)\n"
                        "3. Flag TP53 and BRCA1 specifically in results\n"
                        "4. Write results to tcell_de_results.csv"
                    ),
                    role="data_analyst",
                    max_turns=10,
                ),
            ],
        ),
        # 5. sub: read expression data + metadata
        LLMResponse(
            content=None,
            tool_calls=[
                _tc("s1_sub1_01", "read_file", path="data/tcell_counts.csv"),
                _tc("s1_sub1_02", "read_file", path="data/sample_metadata.csv"),
            ],
        ),
        # 6. sub: run analysis code + write results
        LLMResponse(
            content=None,
            tool_calls=[
                _tc(
                    "s1_sub1_03",
                    "run_code",
                    code=(
                        "import csv, io\n"
                        "# Simulated DESeq2 results (FDR < 0.01)\n"
                        "results = [\n"
                        "    ('TP53', 1.87, 0.0001, 0.0008, 'YES'),\n"
                        "    ('BRCA1', 1.28, 0.0003, 0.002, 'YES'),\n"
                        "    ('MYC', 1.32, 0.0005, 0.003, 'NO'),\n"
                        "    ('JAK1', 1.52, 0.0002, 0.001, 'NO'),\n"
                        "    ('STAT3', 1.41, 0.0004, 0.003, 'NO'),\n"
                        "    ('IL6', 1.68, 0.0001, 0.0008, 'NO'),\n"
                        "    ('TNF', 1.38, 0.0006, 0.004, 'NO'),\n"
                        "]\n"
                        "print(f'DE genes at FDR<0.01: {len(results)}')\n"
                        "print(f'Candidate TP53: log2FC=1.87, FDR=0.0008')\n"
                        "print(f'Candidate BRCA1: log2FC=1.28, FDR=0.002')\n"
                    ),
                    language="python",
                ),
                _tc(
                    "s1_sub1_04",
                    "write_file",
                    filename="tcell_de_results.csv",
                    content=(
                        "gene,log2fc,pvalue,fdr,candidate_flag\n"
                        "TP53,1.87,0.0001,0.0008,YES\n"
                        "BRCA1,1.28,0.0003,0.002,YES\n"
                        "MYC,1.32,0.0005,0.003,NO\n"
                        "JAK1,1.52,0.0002,0.001,NO\n"
                        "STAT3,1.41,0.0004,0.003,NO\n"
                        "IL6,1.68,0.0001,0.0008,NO\n"
                        "TNF,1.38,0.0006,0.004,NO\n"
                    ),
                ),
            ],
        ),
        # 7. sub: terminal completion
        LLMResponse(
            content="Submitting the DE task result.",
            tool_calls=[
                _tc(
                    "s1_sub1_05",
                    "finish_task",
                    summary=(
                        "DE analysis complete. Found 7 genes at FDR < 0.01. "
                        "Candidates flagged: TP53 (log2FC=1.87) and BRCA1 (log2FC=1.28). "
                        "Top gene by fold change: TP53. All 7 genes are immune-related."
                    ),
                    blockers=[],
                    suggestions=["Use the ranked DE table for pathway enrichment."],
                ),
            ],
        ),
        # 8. dispatch volcano plot task
        LLMResponse(
            content="Now creating a volcano plot of the DE results.",
            tool_calls=[
                _tc(
                    "s1_05",
                    "dispatch_subagent",
                    task_id="task_002",
                    stage_id="stage_01",
                    task_description=(
                        "Create a volcano plot from the DE results.\n"
                        "1. Read stages/stage_01/outputs/tcell_de_results.csv\n"
                        "2. Generate volcano plot code (log2FC vs -log10 FDR)\n"
                        "3. Highlight TP53 and BRCA1 as candidate genes\n"
                        "4. Save plot script to volcano_plot.py"
                    ),
                    role="visualization",
                    max_turns=10,
                ),
            ],
        ),
        # 9. sub: read DE results + write plot script
        LLMResponse(
            content=None,
            tool_calls=[
                _tc(
                    "s1_sub2_01",
                    "read_file",
                    path="stages/stage_01/outputs/tcell_de_results.csv",
                ),
                _tc(
                    "s1_sub2_02",
                    "write_file",
                    filename="volcano_plot.py",
                    content=(
                        "import matplotlib.pyplot as plt\n"
                        "import pandas as pd\n"
                        "import numpy as np\n\n"
                        "df = pd.read_csv('tcell_de_results.csv')\n"
                        "df['-log10fdr'] = -np.log10(df['fdr'])\n"
                        "fig, ax = plt.subplots(figsize=(8, 6))\n"
                        "ax.scatter(df['log2fc'], df['-log10fdr'], c='gray', alpha=0.7)\n"
                        "candidates = df[df['candidate_flag'] == 'YES']\n"
                        "ax.scatter(candidates['log2fc'], candidates['-log10fdr'], "
                        "c='red', s=100, label='Candidates')\n"
                        "for _, row in candidates.iterrows():\n"
                        "    ax.annotate(row['gene'], (row['log2fc'], row['-log10fdr']))\n"
                        "ax.set_xlabel('log2 Fold Change')\n"
                        "ax.set_ylabel('-log10 FDR')\n"
                        "ax.set_title('T Cell DE Volcano Plot')\n"
                        "ax.legend()\n"
                        "plt.savefig('volcano_plot.png', dpi=150)\n"
                    ),
                ),
            ],
        ),
        # 10. sub: terminal completion
        LLMResponse(
            content="Submitting the visualization task result.",
            tool_calls=[
                _tc(
                    "s1_sub2_03",
                    "finish_task",
                    summary=(
                        "Volcano plot script saved. TP53 and BRCA1 are highlighted in red. "
                        "The plot shows all 7 DE genes with FDR thresholds."
                    ),
                    blockers=[],
                    suggestions=[],
                ),
            ],
        ),
        # 11. save_conclusion
        LLMResponse(
            content="Concluding stage 1 with DE and visualization results.",
            tool_calls=[
                _tc(
                    "s1_06",
                    "save_conclusion",
                    stage_id="stage_01",
                    content=(
                        "# Stage 1 Conclusion: T Cell Differential Expression\n\n"
                        "## Key Findings\n"
                        "- 7 DE genes at FDR < 0.01 (per user's stricter threshold)\n"
                        "- **TP53**: log2FC = 1.87 (strongest upregulation) — user candidate\n"
                        "- **BRCA1**: log2FC = 1.28 — user candidate\n"
                        "- All 7 genes are immune-related\n\n"
                        "## Artifacts\n"
                        "- tcell_de_results.csv: Full DE table\n"
                        "- volcano_plot.py: Visualization script\n\n"
                        "## Next Steps\n"
                        "Pathway enrichment analysis on the 7 DE genes."
                    ),
                ),
            ],
        ),
        # 12. escalate completed stage 1 findings
        LLMResponse(
            content="Presenting stage 1 findings and asking what to do next.",
            tool_calls=[
                _tc(
                    "s1_07",
                    "escalate_to_user",
                    summary=(
                        "Stage 1 is complete. We found 7 T cell DE genes at FDR < 0.01, "
                        "with TP53 and BRCA1 flagged as requested."
                    ),
                    stage_id="stage_01",
                    artifact_paths=[
                        "stages/stage_01/conclusion.md",
                        "stages/stage_01/outputs/tcell_de_results.csv",
                        "stages/stage_01/outputs/volcano_plot.py",
                    ],
                )
            ],
        ),
    ]


# ── Stage 2: Pathway enrichment ──
# User feedback: full ranked list (not split up/down), add Reactome, highlight JAK-STAT


def _stage2_responses() -> list[LLMResponse]:
    return [
        # 1. save_plan
        LLMResponse(
            content="Planning pathway enrichment on the DE genes from stage 1.",
            tool_calls=[
                _tc(
                    "s2_01",
                    "save_plan",
                    stage_id="stage_02",
                    content=(
                        "# Stage 2: Pathway Enrichment Analysis\n\n"
                        "## Objective\n"
                        "Identify enriched biological pathways among the 7 DE genes.\n\n"
                        "## Method\n"
                        "- GSEA on separated up/down-regulated gene lists\n"
                        "- KEGG pathway database\n"
                    ),
                ),
            ],
        ),
        # 2. escalate
        LLMResponse(
            content="Requesting user guidance on enrichment approach.",
            tool_calls=[
                _tc(
                    "s2_02",
                    "escalate_to_user",
                    summary=(
                        "Plan: Run GSEA on the 7 DE genes, separating up-regulated (6 genes) "
                        "and down-regulated (1 gene). Using KEGG pathways. "
                        "Should I proceed with this approach, or would you prefer a different method?"
                    ),
                    stage_id="stage_02",
                    artifact_paths=[
                        "plans/stage_02_plan.md",
                        "stages/stage_01/outputs/tcell_de_results.csv",
                    ],
                ),
            ],
        ),
        # 3. approve (after user redirects to full ranked list + Reactome)
        LLMResponse(
            content=(
                "User wants full ranked list GSEA (not split by direction), "
                "both KEGG and Reactome, with JAK-STAT highlighted. Approving updated plan."
            ),
            tool_calls=[_tc("s2_03", "approve_plan", stage_id="stage_02")],
        ),
        # 4. dispatch enrichment task
        LLMResponse(
            content="Dispatching GSEA with user's specifications.",
            tool_calls=[
                _tc(
                    "s2_04",
                    "dispatch_subagent",
                    task_id="task_003",
                    stage_id="stage_02",
                    task_description=(
                        "Run GSEA pathway enrichment analysis.\n"
                        "1. Read stages/stage_01/outputs/tcell_de_results.csv for the ranked gene list\n"
                        "2. Run GSEA on the FULL ranked list (do NOT split by direction — user request)\n"
                        "3. Use both KEGG and Reactome databases\n"
                        "4. Specifically highlight JAK-STAT pathway (user's hypothesis)\n"
                        "5. Write results to enrichment_results.csv"
                    ),
                    role="data_analyst",
                    max_turns=10,
                ),
            ],
        ),
        # 5. sub: read DE results
        LLMResponse(
            content=None,
            tool_calls=[
                _tc(
                    "s2_sub1_01",
                    "read_file",
                    path="stages/stage_01/outputs/tcell_de_results.csv",
                ),
            ],
        ),
        # 6. sub: run enrichment + write results
        LLMResponse(
            content=None,
            tool_calls=[
                _tc(
                    "s2_sub1_02",
                    "run_code",
                    code=(
                        "# Simulated GSEA results (KEGG + Reactome)\n"
                        "pathways = [\n"
                        "    ('JAK-STAT signaling', 'KEGG', 0.0001, 0.001, 'JAK1,STAT3,IL6'),\n"
                        "    ('Cytokine-cytokine receptor', 'KEGG', 0.0003, 0.002, 'IL6,TNF'),\n"
                        "    ('NF-kB signaling', 'KEGG', 0.001, 0.006, 'TNF,MYC'),\n"
                        "    ('p53 signaling', 'KEGG', 0.0005, 0.003, 'TP53,BRCA1'),\n"
                        "    ('Immune system', 'Reactome', 0.0001, 0.001, 'JAK1,STAT3,IL6,TNF'),\n"
                        "    ('Signaling by interleukins', 'Reactome', 0.0002, 0.001, 'IL6,JAK1,STAT3'),\n"
                        "    ('DNA repair', 'Reactome', 0.002, 0.01, 'TP53,BRCA1'),\n"
                        "    ('Apoptosis', 'Reactome', 0.003, 0.012, 'TP53,TNF'),\n"
                        "]\n"
                        "print(f'Enriched pathways: {len(pathways)}')\n"
                        "print(f'JAK-STAT signaling: FDR=0.001 (KEGG) — user hypothesis CONFIRMED')\n"
                        "print(f'KEGG pathways: 4, Reactome pathways: 4')\n"
                    ),
                    language="python",
                ),
                _tc(
                    "s2_sub1_03",
                    "write_file",
                    filename="enrichment_results.csv",
                    content=(
                        "pathway,database,pvalue,fdr,genes\n"
                        "JAK-STAT signaling,KEGG,0.0001,0.001,\"JAK1,STAT3,IL6\"\n"
                        "Cytokine-cytokine receptor,KEGG,0.0003,0.002,\"IL6,TNF\"\n"
                        "p53 signaling,KEGG,0.0005,0.003,\"TP53,BRCA1\"\n"
                        "NF-kB signaling,KEGG,0.001,0.006,\"TNF,MYC\"\n"
                        "Immune system,Reactome,0.0001,0.001,\"JAK1,STAT3,IL6,TNF\"\n"
                        "Signaling by interleukins,Reactome,0.0002,0.001,\"IL6,JAK1,STAT3\"\n"
                        "DNA repair,Reactome,0.002,0.01,\"TP53,BRCA1\"\n"
                        "Apoptosis,Reactome,0.003,0.012,\"TP53,TNF\"\n"
                    ),
                ),
            ],
        ),
        # 7. sub: terminal completion
        LLMResponse(
            content="Submitting the enrichment task result.",
            tool_calls=[
                _tc(
                    "s2_sub1_04",
                    "finish_task",
                    summary=(
                        "Enrichment analysis complete. 8 significant pathways found (4 KEGG, 4 Reactome). "
                        "JAK-STAT signaling is the top KEGG hit (FDR=0.001), confirming the user's hypothesis. "
                        "p53 signaling also significant, involving both candidate genes TP53 and BRCA1."
                    ),
                    blockers=[],
                    suggestions=["Compare the enriched pathways with B cell data next."],
                ),
            ],
        ),
        # 8. save_conclusion
        LLMResponse(
            content="Concluding stage 2.",
            tool_calls=[
                _tc(
                    "s2_05",
                    "save_conclusion",
                    stage_id="stage_02",
                    content=(
                        "# Stage 2 Conclusion: Pathway Enrichment\n\n"
                        "## Key Findings\n"
                        "- 8 enriched pathways (4 KEGG + 4 Reactome, per user request)\n"
                        "- **JAK-STAT signaling** (FDR=0.001): User hypothesis confirmed\n"
                        "- p53 signaling involves both candidates TP53 and BRCA1\n"
                        "- Full ranked list GSEA used (not split by direction, per user)\n\n"
                        "## Next Steps\n"
                        "Compare with B cell data to identify cell-type-specific pathways."
                    ),
                ),
            ],
        ),
        # 9. escalate completed stage 2 findings
        LLMResponse(
            content="Presenting stage 2 findings and asking for the next direction.",
            tool_calls=[
                _tc(
                    "s2_06",
                    "escalate_to_user",
                    summary=(
                        "Stage 2 is complete. We found 8 enriched pathways across KEGG and "
                        "Reactome, and JAK-STAT is the top KEGG hit."
                    ),
                    stage_id="stage_02",
                    artifact_paths=[
                        "stages/stage_02/conclusion.md",
                        "stages/stage_02/outputs/enrichment_results.csv",
                    ],
                )
            ],
        ),
    ]


# ── Stage 3: Cross-cell pathway comparison ──
# User feedback: dot plot not heatmap, top 5 shared pathways, add p-value annotations


def _stage3_responses() -> list[LLMResponse]:
    return [
        # 1. save_plan
        LLMResponse(
            content="Planning cross-cell-type comparison between T and B cells.",
            tool_calls=[
                _tc(
                    "s3_01",
                    "save_plan",
                    stage_id="stage_03",
                    content=(
                        "# Stage 3: Cross-Cell Pathway Comparison\n\n"
                        "## Objective\n"
                        "Compare enriched pathways between T cells and B cells.\n\n"
                        "## Method\n"
                        "1. Run DE + enrichment on B cell data (same pipeline as stage 1-2)\n"
                        "2. Identify shared and cell-type-specific pathways\n"
                        "3. Visualize with heatmap\n"
                    ),
                ),
            ],
        ),
        # 2. escalate
        LLMResponse(
            content="Getting user's preferences for the comparison.",
            tool_calls=[
                _tc(
                    "s3_02",
                    "escalate_to_user",
                    summary=(
                        "Plan: Run the same DE+enrichment pipeline on B cells, "
                        "then create a heatmap comparing pathway significance across cell types. "
                        "Should I visualize all pathways or focus on specific ones?"
                    ),
                    stage_id="stage_03",
                    artifact_paths=[
                        "plans/stage_03_plan.md",
                        "stages/stage_02/outputs/enrichment_results.csv",
                    ],
                ),
            ],
        ),
        # 3. approve (after user specifies dot plot, top 5, p-value annotations)
        LLMResponse(
            content=(
                "User wants: dot plot (not heatmap), top 5 shared pathways only, "
                "with p-value annotations. Approving."
            ),
            tool_calls=[_tc("s3_03", "approve_plan", stage_id="stage_03")],
        ),
        # 4. dispatch B cell analysis
        LLMResponse(
            content="First running DE + enrichment on B cell data.",
            tool_calls=[
                _tc(
                    "s3_04",
                    "dispatch_subagent",
                    task_id="task_004",
                    stage_id="stage_03",
                    task_description=(
                        "Run DE + pathway enrichment on B cell data (same pipeline as T cells).\n"
                        "1. Read data/bcell_counts.csv and data/sample_metadata.csv\n"
                        "2. Run DESeq2-style DE analysis (FDR < 0.01)\n"
                        "3. Run GSEA on full ranked list (KEGG + Reactome)\n"
                        "4. Write bcell_de_results.csv and bcell_enrichment.csv"
                    ),
                    role="data_analyst",
                    max_turns=10,
                ),
            ],
        ),
        # 5. sub: read B cell data
        LLMResponse(
            content=None,
            tool_calls=[
                _tc("s3_sub1_01", "read_file", path="data/bcell_counts.csv"),
                _tc("s3_sub1_02", "read_file", path="data/sample_metadata.csv"),
            ],
        ),
        # 6. sub: run analysis + write results
        LLMResponse(
            content=None,
            tool_calls=[
                _tc(
                    "s3_sub1_03",
                    "run_code",
                    code=(
                        "# Simulated B cell DE + enrichment\n"
                        "print('B cell DE genes at FDR<0.01: 5')\n"
                        "print('B cell enriched pathways: 6')\n"
                        "print('Shared with T cell: JAK-STAT, Immune system, Cytokine, p53, NF-kB')\n"
                    ),
                    language="python",
                ),
                _tc(
                    "s3_sub1_04",
                    "write_file",
                    filename="bcell_de_results.csv",
                    content=(
                        "gene,log2fc,pvalue,fdr\n"
                        "BCL2,1.02,0.001,0.005\n"
                        "MYC,0.82,0.003,0.01\n"
                        "JAK1,0.91,0.002,0.008\n"
                        "STAT3,0.88,0.003,0.01\n"
                        "IL6,1.22,0.0005,0.003\n"
                    ),
                ),
                _tc(
                    "s3_sub1_05",
                    "write_file",
                    filename="bcell_enrichment.csv",
                    content=(
                        "pathway,database,pvalue,fdr,genes\n"
                        "JAK-STAT signaling,KEGG,0.001,0.005,\"JAK1,STAT3,IL6\"\n"
                        "Cytokine-cytokine receptor,KEGG,0.002,0.008,\"IL6\"\n"
                        "NF-kB signaling,KEGG,0.005,0.02,\"MYC\"\n"
                        "Immune system,Reactome,0.0005,0.003,\"JAK1,STAT3,IL6\"\n"
                        "Signaling by interleukins,Reactome,0.001,0.005,\"IL6,JAK1,STAT3\"\n"
                        "B cell receptor signaling,Reactome,0.003,0.012,\"BCL2,MYC\"\n"
                    ),
                ),
            ],
        ),
        # 7. sub: terminal completion
        LLMResponse(
            content="Submitting the B cell task result.",
            tool_calls=[
                _tc(
                    "s3_sub1_06",
                    "finish_task",
                    summary=(
                        "B cell analysis complete. 5 DE genes, 6 enriched pathways. "
                        "5 pathways shared with T cells: JAK-STAT, Immune system, "
                        "Cytokine-cytokine receptor, NF-kB, Signaling by interleukins."
                    ),
                    blockers=[],
                    suggestions=["Use the shared pathways for the cross-cell comparison plot."],
                ),
            ],
        ),
        # 8. dispatch comparison visualization
        LLMResponse(
            content="Creating the comparison dot plot per user request.",
            tool_calls=[
                _tc(
                    "s3_05",
                    "dispatch_subagent",
                    task_id="task_005",
                    stage_id="stage_03",
                    task_description=(
                        "Create a dot plot comparing top 5 shared pathways between T and B cells.\n"
                        "1. Read stages/stage_02/outputs/enrichment_results.csv (T cell)\n"
                        "2. Read stages/stage_03/outputs/bcell_enrichment.csv (B cell)\n"
                        "3. Use DOT PLOT (not heatmap — user preference)\n"
                        "4. Show only top 5 shared pathways\n"
                        "5. Add p-value annotations on each dot\n"
                        "6. Save script to comparison_dotplot.py"
                    ),
                    role="visualization",
                    max_turns=10,
                ),
            ],
        ),
        # 9. sub: read both enrichment files
        LLMResponse(
            content=None,
            tool_calls=[
                _tc(
                    "s3_sub2_01",
                    "read_file",
                    path="stages/stage_02/outputs/enrichment_results.csv",
                ),
                _tc(
                    "s3_sub2_02",
                    "read_file",
                    path="stages/stage_03/outputs/bcell_enrichment.csv",
                ),
            ],
        ),
        # 10. sub: write dot plot script
        LLMResponse(
            content=None,
            tool_calls=[
                _tc(
                    "s3_sub2_03",
                    "write_file",
                    filename="comparison_dotplot.py",
                    content=(
                        "import matplotlib.pyplot as plt\n"
                        "import numpy as np\n\n"
                        "# Top 5 shared pathways\n"
                        "pathways = ['JAK-STAT', 'Immune system', 'Cytokine receptor',\n"
                        "            'Interleukin signaling', 'NF-kB']\n"
                        "tcell_fdr = [0.001, 0.001, 0.002, 0.001, 0.006]\n"
                        "bcell_fdr = [0.005, 0.003, 0.008, 0.005, 0.02]\n\n"
                        "fig, ax = plt.subplots(figsize=(10, 6))\n"
                        "y = np.arange(len(pathways))\n"
                        "ax.scatter(-np.log10(tcell_fdr), y, s=100, c='blue', label='T cell')\n"
                        "ax.scatter(-np.log10(bcell_fdr), y, s=100, c='red', label='B cell')\n"
                        "# P-value annotations (user requested)\n"
                        "for i in range(len(pathways)):\n"
                        "    ax.annotate(f'p={tcell_fdr[i]}', (-np.log10(tcell_fdr[i]), y[i]+0.1))\n"
                        "    ax.annotate(f'p={bcell_fdr[i]}', (-np.log10(bcell_fdr[i]), y[i]-0.2))\n"
                        "ax.set_yticks(y)\n"
                        "ax.set_yticklabels(pathways)\n"
                        "ax.set_xlabel('-log10(FDR)')\n"
                        "ax.set_title('Pathway Comparison: T cell vs B cell (Top 5 Shared)')\n"
                        "ax.legend()\n"
                        "plt.tight_layout()\n"
                        "plt.savefig('comparison_dotplot.png', dpi=150)\n"
                    ),
                ),
            ],
        ),
        # 11. sub: terminal completion
        LLMResponse(
            content="Submitting the comparison plot task result.",
            tool_calls=[
                _tc(
                    "s3_sub2_04",
                    "finish_task",
                    summary=(
                        "Dot plot script created. Shows top 5 shared pathways with FDR values "
                        "for both cell types. JAK-STAT is the strongest shared signal."
                    ),
                    blockers=[],
                    suggestions=[],
                ),
            ],
        ),
        # 12. save_conclusion
        LLMResponse(
            content="Concluding the final stage.",
            tool_calls=[
                _tc(
                    "s3_06",
                    "save_conclusion",
                    stage_id="stage_03",
                    content=(
                        "# Stage 3 Conclusion: Cross-Cell Pathway Comparison\n\n"
                        "## Key Findings\n"
                        "- 5 shared pathways between T cells and B cells\n"
                        "- **JAK-STAT signaling** is significant in both (T: FDR=0.001, B: FDR=0.005)\n"
                        "- T cells show stronger pathway enrichment overall\n"
                        "- B cell-specific: B cell receptor signaling pathway\n\n"
                        "## User Preferences Applied\n"
                        "- Dot plot visualization (not heatmap)\n"
                        "- Top 5 shared pathways only\n"
                        "- P-value annotations included\n"
                    ),
                ),
            ],
        ),
        # 13. escalate final stage findings
        LLMResponse(
            content="Presenting the final stage findings to the user.",
            tool_calls=[
                _tc(
                    "s3_07",
                    "escalate_to_user",
                    summary=(
                        "Stage 3 is complete. We identified 5 shared pathways between T and B "
                        "cells, with JAK-STAT as the strongest shared signal."
                    ),
                    stage_id="stage_03",
                    artifact_paths=[
                        "stages/stage_03/conclusion.md",
                        "stages/stage_03/outputs/bcell_enrichment.csv",
                        "stages/stage_03/outputs/comparison_dotplot.py",
                    ],
                )
            ],
        ),
        # 14. terminal tool — main loop completes
        LLMResponse(
            content="Finishing the workflow.",
            tool_calls=[
                _tc(
                    "s3_08",
                    "finish_run",
                    final_summary=(
                        "All three stages complete. Summary:\n"
                        "- Stage 1: 7 T cell DE genes (FDR<0.01), TP53 and BRCA1 confirmed\n"
                        "- Stage 2: 8 enriched pathways, JAK-STAT hypothesis confirmed\n"
                        "- Stage 3: 5 shared pathways with B cells, dot plot generated\n"
                        "Research workflow finished."
                    ),
                )
            ],
        ),
    ]


# ── User responses (domain-specific, influence each stage) ──

USER_RESPONSES = [
    # Initial goal (generated by user_agent.respond at run start)
    (
        "I have T cell and B cell RNA-seq count matrices. "
        "I want to find differentially expressed genes in T cells, "
        "identify enriched pathways, and compare with B cells."
    ),
    # Stage 1: Stricter threshold + candidate genes
    (
        "Use FDR < 0.01 instead of 0.05 — I want stricter filtering for this dataset. "
        "Also, please flag TP53 and BRCA1 specifically in the results, those are our "
        "candidate genes from the preliminary screen. And check whether the DE genes "
        "are immune-related, that's the angle for our paper."
    ),
    # Stage 1 completion: next step
    (
        "Proceed to pathway enrichment next. Use the stage 1 gene ranking you just generated "
        "and keep the focus on immune signaling."
    ),
    # Stage 2: Method and database redirect
    (
        "Don't separate up- and down-regulated genes. Run GSEA on the full ranked list "
        "instead — with only 7 genes the split analysis won't have enough power. "
        "Also add Reactome pathways, not just KEGG. Our PI specifically wants to see "
        "JAK-STAT pathway results, that's our main hypothesis."
    ),
    # Stage 2 completion: next step
    (
        "Now compare the pathway signal against the B cell dataset. I want to know which "
        "signals are shared versus cell-type-specific."
    ),
    # Stage 3: Visualization preferences
    (
        "Good plan, but use a dot plot instead of a heatmap — it's easier to read in the "
        "paper and shows both significance and effect size. Focus on only the top 5 shared "
        "pathways to keep it clean. And add p-value annotations on each dot so reviewers "
        "can see the exact numbers without checking supplementary tables."
    ),
    # Stage 3 completion: stop
    (
        "This is enough for now. Stop here and keep the current results packaged as the final "
        "deliverable for review."
    ),
]


# ── Test ──


def test_three_stage_scenario_with_user_feedback(
    tmp_path: Path,
    real_probe_result: RuntimeProbeResult,
) -> None:
    """Full 3-stage research flow with domain-specific user feedback and mock data."""
    workspace = _setup_workspace(tmp_path)

    # Verify mock data is in place
    assert (workspace.root / "data" / "tcell_counts.csv").is_file()
    assert (workspace.root / "data" / "bcell_counts.csv").is_file()
    assert (workspace.root / "data" / "sample_metadata.csv").is_file()
    assert (workspace.root / "data_catalog.json").is_file()

    all_responses = _stage1_responses() + _stage2_responses() + _stage3_responses()
    scripted_llm = ScriptedLLM(all_responses)
    user_agent = ScriptedUserAgent(list(USER_RESPONSES))

    harness = ResearchHarness(
        model="scripted-scenario",
        workspace=workspace,
        user_agent=user_agent,
        llm_client=scripted_llm,
        system_prompt="You are a research agent for bioinformatics.",
        max_turns=60,
        probe_result=real_probe_result,
    )

    result = harness.run()

    # ── RunResult ──
    assert result.stop_reason == "completed"
    assert result.stages_completed == 3
    assert result.escalation_count == 6
    assert scripted_llm._call_index == len(all_responses), (
        f"Not all responses consumed: {scripted_llm._call_index}/{len(all_responses)}"
    )

    # ── Project state ──
    state = workspace.get_state()
    for stage_id in ("stage_01", "stage_02", "stage_03"):
        assert state.stages[stage_id].stage_status == "completed"
    assert state.current_stage_id == "stage_03"

    # ── Stage 1 artifacts (DE + volcano plot) ──
    s1_outputs = workspace.root / "stages" / "stage_01" / "outputs"
    assert (s1_outputs / "tcell_de_results.csv").is_file()
    assert (s1_outputs / "volcano_plot.py").is_file()
    # Verify DE results contain candidate flags
    de_content = (s1_outputs / "tcell_de_results.csv").read_text()
    assert "TP53" in de_content
    assert "BRCA1" in de_content
    assert "candidate_flag" in de_content

    # ── Stage 2 artifacts (enrichment) ──
    s2_outputs = workspace.root / "stages" / "stage_02" / "outputs"
    assert (s2_outputs / "enrichment_results.csv").is_file()
    enrich_content = (s2_outputs / "enrichment_results.csv").read_text()
    assert "JAK-STAT" in enrich_content
    assert "Reactome" in enrich_content  # user requested Reactome

    # ── Stage 3 artifacts (B cell + comparison) ──
    s3_outputs = workspace.root / "stages" / "stage_03" / "outputs"
    assert (s3_outputs / "bcell_de_results.csv").is_file()
    assert (s3_outputs / "bcell_enrichment.csv").is_file()
    assert (s3_outputs / "comparison_dotplot.py").is_file()
    # Verify dot plot (not heatmap, per user request)
    dotplot_content = (s3_outputs / "comparison_dotplot.py").read_text()
    assert "dot" in dotplot_content.lower() or "scatter" in dotplot_content.lower()
    assert "annotate" in dotplot_content  # p-value annotations per user request

    # ── Task results ──
    task_files = [
        ("stage_01", "task_001"),
        ("stage_01", "task_002"),
        ("stage_02", "task_003"),
        ("stage_03", "task_004"),
        ("stage_03", "task_005"),
    ]
    for stage_id, task_id in task_files:
        path = workspace.root / "stages" / stage_id / "tasks" / f"{task_id}_result.json"
        assert path.is_file(), f"Missing {path}"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["status"] == "success"

    # ── Session log ──
    session_path = workspace.root / "session.json"
    assert session_path.is_file()
    session = json.loads(session_path.read_text(encoding="utf-8"))
    assert session["run_result"]["stages_completed"] == 3
    assert session["project_state"]["stages"]["stage_03"]["stage_status"] == "completed"
    assert len(session["subagent_runs"]) == 5
    roles = [r["role"] for r in session["subagent_runs"]]
    assert roles == ["data_analyst", "visualization", "data_analyst", "data_analyst", "visualization"]
