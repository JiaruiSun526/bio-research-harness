"""Generate evaluation_checklist.json from a scientific paper.

Usage:
    python scripts/generate_checklist.py \
        --paper references/user_paper/paper_text.txt \
        --data-dir references/user_paper/data \
        --output evaluation_checklist.json \
        [--model openrouter/openai/gpt-5.4]

Input:
    - Paper (required): .pdf or .txt, sent as multimodal content to the LLM.
    - Data directory (optional): scanned for available data files so the LLM
      can reference real paths in output_pattern fields.

Output:
    - evaluation_checklist.json — structured checklist of verifiable claims
      organized by proof flows, used as ground truth for automated evaluation.

The checklist is generated offline from the paper and is independent of both
research_brief.md and the agent's runtime workspace.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from prepare_scenario import (
    PROJECT_ROOT,
    _build_client,
    _build_paper_content,
    _load_openrouter_config,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt: instruct LLM to extract verifiable claims as a structured checklist.
# Copied verbatim from scripts_for_assets/generate_eval_checklist.py.
# ---------------------------------------------------------------------------

CHECKLIST_PROMPT = """\
You are a scientific paper analyst specialized in reproducibility assessment. \
Your task is to read a research paper and produce a structured evaluation \
checklist in JSON format. This checklist will be used by an automated system \
to score how well a computational reproduction matches the original paper.

## Core concept: proof flows

A "proof flow" is the paper's argument chain from raw data to a specific \
conclusion. Each major claim the paper makes — typically culminating in a \
figure, table, or key statistic — anchors one proof flow. Organize your \
checklist around these flows (not by figure number).

A proof flow typically has:
1. Input data → preprocessing → analysis method → output artifact → claim

Proof flow granularity guidance:
- Do NOT create a proof flow for a single preparatory step that has no \
standalone claim (e.g., "download gene list" is not a proof flow — it is \
a pipeline item within the flow that uses that gene list).
- Do NOT merge unrelated claims into one flow just because they share input \
data (e.g., survival analysis and functional enrichment both use DEGs but \
make independent claims — they are separate flows).

## JSON schema

Return a SINGLE JSON object with exactly this structure:

{
  "paper": {
    "title": "<exact paper title>",
    "doi": "<DOI or URL, or null>"
  },
  "proof_flows": [
    {
      "id": "pf<N>",
      "name": "<one sentence describing this argument chain>",
      "items": [
        {
          "id": "pf<N>.<M>",
          "priority": "critical | important | supplementary",
          "check_type": "pipeline | qualitative | quantitative",
          "verify": "<what to verify>",
          "paper_reference": "<Fig.X / Table Y / Section Z>",
          "expected": "<expected result, with numbers if quantitative>",
          "tolerance": "<e.g. ±5%, ±0.02, or null if not quantitative>",
          "output_pattern": "<workspace glob pattern for expected output file>"
        }
      ]
    }
  ],
  "scoring": {
    "weights": {"critical": 3, "important": 2, "supplementary": 1},
    "verdicts": {"pass": 1.0, "partial": 0.5, "fail": 0.0}
  }
}

## check_type definitions

All three check_types MUST be used in a well-formed checklist. A proof flow \
that only has quantitative items without a pipeline item for its prerequisite \
data/artifact is incomplete.

- **pipeline**: The analysis step ran successfully and produced a usable \
output artifact. Verified by checking file existence, non-empty content, \
and correct format. Every proof flow SHOULD begin with at least one pipeline \
item that confirms the prerequisite data or intermediate result exists \
before quantitative checks are applied. Pipeline items are the foundation — \
without them, downstream quantitative checks are meaningless.
- **qualitative**: The output matches the paper's description in kind, \
pattern, or direction — but not exact numbers. Use when the paper reports \
trends, rankings, or visual patterns (e.g., "top 5 enriched pathways include \
immune response", "heatmap shows two distinct clusters").
- **quantitative**: The output matches a specific number from the paper \
within a tolerance. Use when the paper gives exact values (e.g., "AUC = 0.78", \
"342 differentially expressed genes at FDR < 0.05"). Always include \
tolerance for quantitative items.

## Priority guidelines

- **critical**: Main results — claims in the abstract, key figures, primary \
tables. Failure here means the reproduction is fundamentally wrong.
- **important**: Supporting results — secondary figures, supplementary \
analyses that strengthen the main claims.
- **supplementary**: Nice-to-have — formatting, exact visual similarity, \
minor sub-analyses.

## Rules

1. Every Figure and Table in the paper MUST have at least one checklist item. \
Use `paper_reference` to identify the specific figure panel or table \
(e.g. 'Fig.2A' not just 'Fig.2').
2. For quantitative items, extract the EXACT value from the paper and set a \
tolerance appropriate to the nature of the quantity:
   - Deterministic counts (sample sizes, gene counts after filtering, group \
sizes) are either correct or wrong — use "±0".
   - Statistical point estimates (hazard ratios, correlation coefficients, \
AUC) depend on implementation details — use ±5% or a small absolute margin \
(e.g., "±0.05" for an HR of 1.22).
   - P-values are sensitive to implementation differences — use a tolerance \
that preserves the significance conclusion. For p < 0.01, ±50% relative \
is reasonable (e.g., p=0.00011 tolerates 0.00005–0.00017). For p near 0.05, \
use ±0.01 absolute.
   - Proportions/percentages derived from deterministic counts inherit "±0".
   If the paper does not give a number, use qualitative instead.
3. The "expected" field must be self-contained — an evaluator should \
understand what to look for without reading the paper.
4. output_pattern must use glob syntax relative to the workspace root. \
The workspace structure is: `stages/<stage_id>/outputs/<filename>`. \
All agent outputs (CSV, PNG, JSON, MD, etc.) are written to stage output \
directories. Use patterns like `stages/*/outputs/*calibration*.csv`, \
`stages/*/outputs/*fig2*.png`, `stages/*/outputs/*.json`. \
Do NOT use `results/` or `figures/` — those directories do not exist.
5. The "scoring" field must be EXACTLY as shown above — do not modify it.
6. Extract ONLY values explicitly stated in the paper's text, figures, or \
tables. Do NOT compute derived values (e.g., percentages from raw counts, \
differences between reported numbers) unless those derived values also \
appear explicitly in the paper. If a value comes from a figure annotation \
rather than the text, note this in the "expected" field.
7. Aim for 10-40 items total across all proof flows.
8. If the paper reports conflicting values for the same result in different \
places (e.g., text says p=0.028 but figure shows p=0.0015), include BOTH \
values in the "expected" field with their sources, and use the more \
conservative (larger) tolerance to accommodate either value. Flag the \
discrepancy explicitly so the evaluator is aware.
9. If a reported value appears inconsistent with the paper's own \
interpretation (e.g., a non-significant p-value described as indicating \
strong significance), flag this in the "expected" field as a likely error \
and note what the correct value probably is based on context. Do not \
silently accept values that contradict their surrounding narrative.
10. check_type must match the content of "expected": if "expected" contains \
specific numbers (correlation coefficients, p-values, counts), the item \
MUST be quantitative with an appropriate tolerance — do not label it \
qualitative. Use qualitative ONLY when "expected" describes patterns, \
directions, or rankings without exact values.

{data_inventory}
"""


# ---------------------------------------------------------------------------
# Data inventory scanning (from scripts_for_assets/generate_eval_checklist.py)
# ---------------------------------------------------------------------------


def _scan_data_inventory(data_dir: Path | None) -> str:
    """Build a text inventory of available data files for the prompt."""
    if data_dir is None or not data_dir.exists():
        return ""

    lines = ["Available data files (already downloaded):"]
    for root, _dirs, files in sorted(os.walk(data_dir)):
        rel = Path(root).relative_to(data_dir)
        for fname in sorted(files):
            fpath = Path(root) / fname
            size_mb = fpath.stat().st_size / (1024 * 1024)
            if size_mb >= 0.01:
                lines.append(f"  {rel / fname}  ({size_mb:.1f} MB)")
            else:
                lines.append(f"  {rel / fname}  ({fpath.stat().st_size} bytes)")
    return "\n".join(lines) if len(lines) > 1 else ""


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------


def generate_checklist_from_paper(
    paper_path: Path,
    data_dir: Path | None,
    client: OpenAI,
    model: str,
) -> str:
    """Call LLM with paper content + prompt, return pretty-printed JSON string.

    Uses _build_paper_content from prepare_scenario to handle both .pdf and
    .txt papers as multimodal content.
    """
    data_inventory = _scan_data_inventory(data_dir)
    inventory_block = (
        f"\nThe following data files are already available locally — use them "
        f"to inform output_pattern fields:\n```\n{data_inventory}\n```"
        if data_inventory
        else ""
    )

    system_prompt = CHECKLIST_PROMPT.replace("{data_inventory}", inventory_block)

    # Build paper content using prepare_scenario's multimodal handler
    paper_content = _build_paper_content(paper_path)
    user_content: list[dict[str, Any]] = paper_content + [
        {
            "type": "text",
            "text": (
                "Please read the paper above and generate an evaluation "
                "checklist following the JSON schema in the system prompt.\n\n"
                "Output ONLY the JSON object. No preamble, no code fences."
            ),
        },
    ]

    log.info("Calling LLM model=%s with paper (%s)...", model, paper_path.name)
    messages = cast(
        list[ChatCompletionMessageParam],
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    create_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": 32768,
        "temperature": 0.2,
    }
    if "openai/" in model or "gpt" in model.lower():
        create_kwargs["extra_body"] = {"reasoning_effort": "high"}

    # Try with response_format first; fall back without it for models that
    # don't support structured output on OpenRouter (e.g. Anthropic models).
    content = ""
    try:
        response = client.chat.completions.create(
            **create_kwargs, response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content or ""
    except Exception:
        log.warning("response_format not supported; retrying without it")

    if not content.strip():
        # Reasoning models may exhaust tokens on thinking. Retry without
        # reasoning_effort and response_format.
        log.warning("Empty response; retrying without reasoning_effort")
        retry_kwargs = {k: v for k, v in create_kwargs.items() if k != "extra_body"}
        response = client.chat.completions.create(**retry_kwargs)
        content = response.choices[0].message.content or ""

    log.info("LLM response received (%d chars)", len(content))

    # Strip markdown code fences if the model wrapped the JSON.
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        lines = lines[1:]  # drop ```json
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines)

    parsed = json.loads(stripped)
    return json.dumps(parsed, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate evaluation_checklist.json from a scientific paper",
    )
    parser.add_argument(
        "--paper", type=Path, required=True,
        help="Path to the paper (.pdf or .txt)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Path to the downloaded data directory (for file inventory)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation_checklist.json"),
        help="Output path (default: evaluation_checklist.json)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="LLM model (default: from config/llm.toml [openrouter].default_model)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if not args.paper.exists():
        log.error("Paper not found: %s", args.paper)
        sys.exit(1)

    # Load config and build client
    config = _load_openrouter_config()
    model = args.model or config["default_model"]
    if model.startswith("openrouter/"):
        model = model[len("openrouter/"):]
    client = _build_client(config["api_key"], config["proxy"])

    checklist_json = generate_checklist_from_paper(
        paper_path=args.paper,
        data_dir=args.data_dir,
        client=client,
        model=model,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(checklist_json)
    log.info("Evaluation checklist written to: %s", args.output)
    print(
        f"Generated {args.output} ({len(checklist_json)} chars)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

