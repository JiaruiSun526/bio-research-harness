"""Run a configurable validation scenario with SimulatedUser.

Reads scenario configuration from a TOML file and runs the research harness
with the specified data, research brief, and initial message. This generalizes
run_scenario_live.py to support arbitrary validation scenarios.

Usage: python scripts/run_validation.py <scenario.toml>
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import tomli
except ModuleNotFoundError:
    import tomllib as tomli  # type: ignore[no-redef]

from research_agent.harness import ResearchHarness
from research_agent.llm_client import LLMClient
from research_agent.user_agent.simulated import SimulatedUser
from research_agent.workspace import Workspace

LLM_CONFIG_PATH = PROJECT_ROOT / "config" / "llm.toml"


def load_scenario(scenario_path: Path) -> dict:
    """Load and return parsed TOML scenario config.

    Validates that required top-level keys exist. Lets tomli raise
    on malformed TOML.
    """
    raw = scenario_path.read_text(encoding="utf-8")
    config = tomli.loads(raw)

    required_keys = ["name", "data", "simulated_user", "run"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        print(f"ERROR: Scenario file missing required keys: {missing}", file=sys.stderr)
        sys.exit(1)

    if "sources" not in config["data"]:
        print("ERROR: Scenario [data] section missing 'sources'", file=sys.stderr)
        sys.exit(1)
    if "catalog" not in config["data"]:
        print("ERROR: Scenario [data] section missing 'catalog'", file=sys.stderr)
        sys.exit(1)
    if "research_brief" not in config["simulated_user"]:
        print("ERROR: Scenario [simulated_user] section missing 'research_brief'", file=sys.stderr)
        sys.exit(1)
    # initial_message is optional (backward compat); SimulatedUser generates it at runtime

    return config


def setup_workspace(scenario_name: str, data_sources: list[str], data_catalog: list[dict]) -> Workspace:
    """Create timestamped workspace, copy data files, and write data_catalog.json.

    Data source paths are resolved relative to PROJECT_ROOT.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_dir = (PROJECT_ROOT / "runs" / f"{scenario_name}_{timestamp}").resolve()

    workspace = Workspace(workspace_dir)
    workspace.initialize()

    data_dir = workspace.root / "data"
    data_dir.mkdir(exist_ok=True)

    for source in data_sources:
        source_path = (PROJECT_ROOT / source).resolve()
        if not source_path.is_file():
            print(f"ERROR: Data source file not found: {source_path}", file=sys.stderr)
            sys.exit(1)
        # Preserve subdirectory structure under the "data/" segment of the
        # source path so species-specific files don't collide.
        # e.g. "papers/conn2res/data/drosophila/conn.csv" → "data/drosophila/conn.csv"
        parts = Path(source).parts
        if "data" in parts:
            rel = Path(*parts[parts.index("data") + 1 :])
        else:
            rel = Path(source_path.name)
        dest = data_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_path, dest)
        print(f"  Copied {source} -> data/{rel}")

    catalog_json = json.dumps({"datasets": data_catalog}, indent=2)
    (workspace.root / "data_catalog.json").write_text(catalog_json, encoding="utf-8")
    print(f"  Wrote data_catalog.json ({len(data_catalog)} datasets)")

    # Copy skill reference files into workspace so sub-agents can read them
    skills_src = PROJECT_ROOT / "skills"
    if skills_src.is_dir():
        skills_dst = workspace.root / "skills"
        shutil.copytree(skills_src, skills_dst, dirs_exist_ok=True)
        skill_count = sum(1 for _ in skills_dst.glob("*/SKILL.md"))
        print(f"  Copied skills/ ({skill_count} skill guides)")

    return workspace


def load_llm_config(provider: str = "mimo") -> dict:
    """Load LLM config from config/llm.toml [provider] section."""
    if not LLM_CONFIG_PATH.is_file():
        print(f"ERROR: LLM config not found: {LLM_CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)

    config = tomli.loads(LLM_CONFIG_PATH.read_text(encoding="utf-8"))
    if provider not in config:
        print(f"ERROR: LLM config missing [{provider}] section", file=sys.stderr)
        sys.exit(1)

    section = config[provider]
    required = ["default_model", "api_key"]
    missing = [k for k in required if k not in section]
    if missing:
        print(f"ERROR: [{provider}] section missing required keys: {missing}", file=sys.stderr)
        sys.exit(1)

    return section


def parse_args() -> tuple[Path, Path | None, str, str | None]:
    """Parse CLI arguments.

    Returns (scenario_path, resume_workspace_path, provider, user_provider).

    Usage:
        python scripts/run_validation.py <scenario.toml>
        python scripts/run_validation.py <scenario.toml> --provider openrouter
        python scripts/run_validation.py <scenario.toml> --provider openrouter --user-provider mimo
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Run or resume a validation scenario with SimulatedUser.",
    )
    parser.add_argument(
        "scenario",
        type=Path,
        help="Path to the scenario TOML file.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="WORKSPACE_PATH",
        help="Resume a previous run from the given workspace directory.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="mimo",
        help="LLM provider for main agent + sub-agents (default: mimo)",
    )
    parser.add_argument(
        "--user-provider",
        type=str,
        default=None,
        help="Separate LLM provider for SimulatedUser (default: same as --provider)",
    )
    args = parser.parse_args()

    scenario_path = args.scenario.resolve()
    if not scenario_path.is_file():
        parser.error(f"Scenario file not found: {scenario_path}")

    resume_path: Path | None = None
    if args.resume is not None:
        resume_path = args.resume.resolve()
        if not resume_path.is_dir():
            parser.error(f"Resume workspace not found: {resume_path}")
        if not (resume_path / "session.json").is_file():
            parser.error(f"No session.json in resume workspace: {resume_path}")

    return scenario_path, resume_path, args.provider, args.user_provider


def _make_client(llm_cfg: dict) -> tuple[str, LLMClient]:
    """Build model identifier and LLMClient from a config section."""
    model_name = llm_cfg["default_model"]
    if model_name.startswith("openrouter/"):
        model_name = model_name[len("openrouter/"):]
    model = f"openai/{model_name}"
    client = LLMClient(
        default_model=model,
        proxy=llm_cfg.get("proxy") or None,
        api_base=llm_cfg.get("api_base"),
        api_key=llm_cfg["api_key"],
    )
    return model, client


def main() -> None:
    scenario_path, resume_path, provider, user_provider = parse_args()

    # Load scenario config
    print(f"Loading scenario: {scenario_path}")
    scenario = load_scenario(scenario_path)
    scenario_name = scenario["name"]
    max_turns = scenario["run"].get("max_turns", 50)

    # Load LLM config — agent
    print(f"Loading LLM config (agent: {provider})...")
    agent_cfg = load_llm_config(provider)
    model, client = _make_client(agent_cfg)

    # Load LLM config — user (separate provider if specified)
    if user_provider and user_provider != provider:
        print(f"Loading LLM config (user: {user_provider})...")
        user_cfg = load_llm_config(user_provider)
        user_model, user_client = _make_client(user_cfg)
    else:
        user_model, user_client = model, client

    if resume_path is not None:
        # Resume path: reuse existing workspace, skip data setup
        print(f"Resuming from workspace: {resume_path}")
        workspace = Workspace(resume_path)

        user_agent = SimulatedUser(
            research_brief=scenario["simulated_user"]["research_brief"],
            llm_client=user_client,
            model=user_model,
            workspace_root=workspace.root,
        )

        harness = ResearchHarness(
            model=model,
            workspace=workspace,
            user_agent=user_agent,
            llm_client=client,
            max_turns=max_turns,
        )

        print(f"\nResuming validation scenario: {scenario_name}")
        print(f"Agent model: {model}")
        if user_model != model:
            print(f"User model:  {user_model}")
        print(f"Max turns: {max_turns}")
        print(f"Workspace: {workspace.root}\n")

        result = harness.resume(resume_path)

    else:
        # Fresh run path: create workspace, copy data, run from scratch
        print(f"Setting up workspace for scenario '{scenario_name}'...")
        workspace = setup_workspace(
            scenario_name,
            scenario["data"]["sources"],
            scenario["data"]["catalog"],
        )

        user_agent = SimulatedUser(
            research_brief=scenario["simulated_user"]["research_brief"],
            llm_client=user_client,
            model=user_model,
            workspace_root=workspace.root,
        )

        harness = ResearchHarness(
            model=model,
            workspace=workspace,
            user_agent=user_agent,
            llm_client=client,
            max_turns=max_turns,
        )

        print(f"\nStarting validation scenario: {scenario_name}")
        print(f"Agent model: {model}")
        if user_model != model:
            print(f"User model:  {user_model}")
        print(f"Max turns: {max_turns}")
        print(f"Workspace: {workspace.root}\n")

        result = harness.run()

    print(f"\n{'='*60}")
    print(f"RUN COMPLETE — {scenario_name}")
    print(f"Stop reason: {result.stop_reason}")
    print(f"Stages completed: {result.stages_completed}")
    print(f"Tool calls: {result.tool_call_count}")
    print(f"Escalations: {result.escalation_count}")
    print(f"Tokens: {result.prompt_tokens} prompt + {result.completion_tokens} completion")
    print(f"Session log: {workspace.root / 'session.json'}")
    print(f"{'='*60}")
    print(f"\nView: streamlit run src/research_agent/viewer.py -- {workspace.root / 'session.json'}")


if __name__ == "__main__":
    main()
