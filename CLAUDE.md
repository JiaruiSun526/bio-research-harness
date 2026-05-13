# Research Agent

Multi-agent research system: main coordination agent + sub agents for execution, with human or simulated user in the loop.

## Key Paths

- `src/research_agent/` — core modules
- `tests/` — unit, integration, scenario, smoke tests
- `config/llm.toml` — LLM provider config (gitignored), see `config/llm.toml.example`
- `tests/fixtures/` — mock expression matrices and data catalog

## Commands

- Tests: `python -m pytest tests/ -v --ignore=tests/test_smoke_api.py`
- Smoke test: `python -m pytest tests/test_smoke_api.py -v`
- Live run: `python scripts/run_scenario_live.py`
- Viewer: `streamlit run src/research_agent/viewer.py -- <session.json>`

## Sub-Agent Runtime Environment

The sub-agent code execution sandbox supports **Python** and optionally **R**.

### Python
**Required** (always present): numpy, pandas, scipy, matplotlib
**Optional** (present when installed): seaborn, statsmodels, sklearn, gseapy, pydeseq2, adjustText, xgboost, shap, lightgbm, emcee, lmfit, optuna, plotly, umap, alphastats, torch, networkx, sympy, natsort, joblib, xlsxwriter, corner, ddeint, bct (bctpy), reservoirpy, inmoose, matplotlib_venn, pimmslearn, tellurium, rrplugins, sobol, sobol_seq, halo

### R (optional)
R support is auto-probed at startup. When Rscript is found, sub-agents can use `run_code(code, language="r")`. When not found, `language="r"` is structurally rejected.

**All R packages are optional** — probed from `DEFAULT_R_OPTIONAL_PACKAGES` in runtime_env.py. Includes tidyverse core, Bioconductor DE/pathway/enrichment, single-cell, survival, and general stats packages.

### Unavailable packages
**NOT available (Python)**: graph-tool (requires system libs/conda), or any package not listed above.

Source of truth: `src/research_agent/runtime_env.py` (`DEFAULT_REQUIRED_PACKAGES`, `DEFAULT_OPTIONAL_PACKAGES`, `DEFAULT_R_OPTIONAL_PACKAGES`).

When preparing scenarios (`scripts/prepare_scenario.py`), the BRIEF_PROMPT includes the runtime package list so the LLM generates briefs that only reference available packages.

## Agent Engineering Principles

These principles come from observing real LLM behavior in live runs. They apply to this project and to multi-agent system design in general.

### Harness over Prompt

When an agent must not do X, make X structurally impossible — don't just tell it "please don't do X". Prompt instructions are suggestions; harness constraints are laws.

- If files must be persisted, make the execution environment persist them (e.g. set cwd). Don't rely on agents choosing the right tool.
- If a language/package is unavailable, remove it from the tool, don't just say "don't use it" in the prompt.
- If a workflow step is commonly forgotten, embed a nudge in the preceding tool's return value.

### Context at the Point of Need

Agents forget system prompt instructions as conversation grows. Provide critical information where it's consumed, not where it's declared.

- Inject workspace file listings into sub agent context at dispatch time, not in a system prompt they received 20 turns ago.
- Include environment capabilities (available packages, versions) in tool descriptions or task context, not just the system prompt.
- Validate and correct agent-provided paths at the harness level rather than hoping the agent types them correctly.

### Diagnose Root Cause, Don't Patch Symptoms

When an agent misbehaves, ask: **what information was the agent missing at the point of decision?** The fix is to supply that information through context (tool results, injected state), not to add more instructions.

Example: agent writes `to_csv('stages/.../outputs/file.csv')` creating nested paths. The symptom looks like "agent used wrong path format". But the root cause is **the agent didn't know its cwd**. Adding "use relative paths" to the prompt is a patch. Showing `[cwd: stages/stage_01/outputs/]` in every tool result is a fix — the agent now has the information to make the right decision itself.

The diagnostic question is always: "if I were the agent, and I only saw what the agent sees (system prompt + message history + tool results), would I have enough information to do the right thing?" If not, the fix is context engineering, not prompt engineering.

### Structural Constraints > Behavioral Instructions

When prompt says "do A not B" but the agent's training strongly associates the task with B, the agent will try B first. The cost is wasted turns and retries.

- Remove unsupported options from tool interfaces entirely.
- Auto-inject boilerplate (imports, setup) that agents consistently forget.
- Make success paths easy and failure paths loudly informative.

### User-Driven Stage Progression

The user drives research direction. The agent executes, not decides.

- Stages are independent work units (plan + tasks + conclusion), not a linear pipeline.
- The agent does NOT decide what comes next. After completing a stage, the agent presents results to the user and waits for the user's next instruction.
- Stage completion is a derived state (has conclusion + all tasks returned), not an explicit agent action. There is no "advance" operation.
- The user may go back, branch, or change direction at any point. The system must support non-linear stage graphs.

### SimulatedUser Needs Behavioral Directives

A one-sentence role instruction ("act as the user") produces unpredictable behavior. The simulated user needs explicit guidance on:
- When to approve vs push back
- Response length and specificity expectations
- Whether to introduce concerns beyond the research brief
- How to signal approval unambiguously
