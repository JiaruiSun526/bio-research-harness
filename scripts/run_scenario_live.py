"""Run a live scenario with real LLM (MiMo) and SimulatedUser.

Both the main agent and the simulated user are driven by MiMo.
The simulated user is given a research brief describing goals and preferences,
and responds dynamically to whatever the agent asks.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import tomli
except ModuleNotFoundError:
    import tomllib as tomli  # type: ignore[no-redef]

from research_agent.harness import ResearchHarness
from research_agent.llm_client import LLMClient
from research_agent.user_agent.simulated import SimulatedUser
from research_agent.workspace import Workspace

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "llm.toml"
FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

RESEARCH_BRIEF = """\
# Research Brief: T Cell and B Cell Differential Expression Comparison

## Background
You are a computational biologist studying immune cell responses to treatment.
You have RNA-seq count matrices for T cells and B cells (10 genes x 6 samples
each: 3 control, 3 treatment).

## Research Goals
1. Find differentially expressed genes in T cells
2. Run pathway enrichment analysis on the DE results
3. Compare enriched pathways between T cells and B cells

## Your Preferences and Domain Knowledge
- You prefer strict statistical thresholds (FDR < 0.01) because the dataset
  is small and you want to be conservative for publication.
- TP53 and BRCA1 are your candidate genes from a preliminary screen. You want
  them flagged in the DE results.
- For pathway enrichment, you prefer GSEA on the full ranked gene list (not
  split by up/down regulation) because splitting a small gene set loses power.
- You want both KEGG and Reactome databases used for enrichment.
- JAK-STAT signaling is your main biological hypothesis — if it comes up
  significant, that strengthens the paper.
- For comparison visualizations, you prefer dot plots over heatmaps because
  they're easier to read in papers. Focus on top 5 shared pathways only.
- You want p-value annotations on plots so reviewers don't need supplementary tables.

## Communication Style
- Give specific, actionable feedback when the agent presents plans.
- Point out when you disagree with methodological choices.
- Approve plans when they incorporate your feedback adequately.
- Keep responses concise (2-4 sentences).
"""


def main() -> None:
    from datetime import datetime

    config = tomli.loads(CONFIG_PATH.read_text())
    mimo_cfg = config["mimo"]

    # Workspace with timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_dir = Path(f"runs/live_{timestamp}").resolve()

    workspace = Workspace(workspace_dir)
    workspace.initialize()

    data_dir = workspace.root / "data"
    data_dir.mkdir()
    shutil.copy(FIXTURES_DIR / "data_catalog.json", workspace.root / "data_catalog.json")
    for f in ("tcell_counts.csv", "bcell_counts.csv", "sample_metadata.csv"):
        shutil.copy(FIXTURES_DIR / f, data_dir / f)

    # LLM client (shared by both agent and simulated user)
    model = f"openai/{mimo_cfg['default_model']}"
    client = LLMClient(
        default_model=model,
        proxy=mimo_cfg.get("proxy") or None,
        api_base=mimo_cfg.get("api_base"),
        api_key=mimo_cfg["api_key"],
    )

    # SimulatedUser driven by same LLM
    user_agent = SimulatedUser(
        research_brief=RESEARCH_BRIEF,
        llm_client=client,
        model=model,
        workspace_root=workspace.root,
    )

    harness = ResearchHarness(
        model=model,
        workspace=workspace,
        user_agent=user_agent,
        llm_client=client,
        max_turns=50,
    )

    print(f"Starting live scenario with SimulatedUser...")
    print(f"Model: {model}")
    print(f"Workspace: {workspace_dir}\n")

    result = harness.run(
        "I have T cell and B cell RNA-seq count matrices (10 genes x 6 samples each, "
        "3 control + 3 treatment). I want to:\n"
        "1. Find differentially expressed genes in T cells\n"
        "2. Run pathway enrichment on the DE results\n"
        "3. Compare enriched pathways between T cells and B cells\n\n"
        "Start by reading the data catalog and planning the first stage."
    )

    print(f"\n{'='*60}")
    print(f"RUN COMPLETE")
    print(f"Stop reason: {result.stop_reason}")
    print(f"Stages completed: {result.stages_completed}")
    print(f"Tool calls: {result.tool_call_count}")
    print(f"Escalations: {result.escalation_count}")
    print(f"Tokens: {result.prompt_tokens} prompt + {result.completion_tokens} completion")
    print(f"Session log: {workspace_dir / 'session.json'}")
    print(f"{'='*60}")
    print(f"\nView: streamlit run src/research_agent/viewer.py -- {workspace_dir / 'session.json'}")


if __name__ == "__main__":
    main()
