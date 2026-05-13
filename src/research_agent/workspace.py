"""Workspace filesystem manager — the project's system of record.

All write operations maintain consistency of project_state.json.
Read operations support progressive disclosure (agent reads on demand).

Workspace directory layout:
    workspace/
    ├── project_state.json          # harness-maintained global state
    ├── data_catalog.json           # injected at startup, main agent context
    ├── plans/
    │   └── stage_XX_plan.md        # approved plans (Markdown)
    └── stages/
        └── stage_XX/
            ├── conclusion.md       # stage conclusion written by main agent
            ├── tasks/
            │   └── task_XXX_result.json   # subagent structured results
            └── outputs/            # subagent-generated artifacts
"""

from __future__ import annotations

import itertools
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .schemas import ProjectState, StageState, StageStatus, TaskResult


logger = logging.getLogger(__name__)

_LARGE_FILE_THRESHOLD: int = 51_200  # 50KB — files above this get structured preview
_TABULAR_EXTENSIONS: frozenset[str] = frozenset({".csv", ".tsv", ".xlsx"})


def _build_binary_placeholder(file_path: Path, relative_path: str, file_size: int) -> str:
    """Return a placeholder for binary files that cannot be displayed as text.

    Agents need to know the file exists and its size, but raw binary content
    is useless in an LLM context window. This gives them enough to reference
    the file in run_code calls.
    """
    size_kb = file_size / 1024
    return (
        f"[Binary file: {relative_path}]\n"
        f"Size: {size_kb:.1f} KB\n"
        f"Cannot display binary content. Use this path to reference the file."
    )


def _build_tabular_preview(file_path: Path, relative_path: str) -> str:
    """Return a structured preview of a CSV/TSV data file using stdlib csv.

    Large data files blow up the LLM context if returned in full. This gives
    agents what they need to write correct pandas code: column count, column
    names, row count, and a few sample rows — all without importing pandas
    (which is a runtime dependency, not available in the workspace's venv).

    For .xlsx files, falls back to text preview since openpyxl/pandas aren't
    guaranteed to be available.
    """
    import csv

    suffix = file_path.suffix.lower()
    if suffix == ".xlsx":
        # Can't parse Excel without pandas/openpyxl — treat as binary
        return _build_binary_placeholder(
            file_path, relative_path, file_path.stat().st_size
        )

    delimiter = "\t" if suffix == ".tsv" else ","

    try:
        with open(file_path, encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            header = next(reader, None)
            if header is None:
                return f"[Empty data file: {relative_path}]"
            sample_rows: list[list[str]] = []
            for row in itertools.islice(reader, 5):
                sample_rows.append(row)
    except (UnicodeDecodeError, csv.Error) as exc:
        logger.warning("Failed to parse tabular file %s: %s", relative_path, exc)
        size_mb = file_path.stat().st_size / (1024 * 1024)
        return (
            f"[Data file: {relative_path}]\n"
            f"Size: {size_mb:.1f} MB\n"
            f"Could not parse ({type(exc).__name__}: {exc}).\n"
            f"Use run_code with pandas to inspect this file."
        )

    # Count rows cheaply via raw byte-level line scan
    with open(file_path, "rb") as fh:
        total_rows = sum(1 for _ in fh) - 1  # subtract header

    total_cols = len(header)
    size_mb = file_path.stat().st_size / (1024 * 1024)

    # Column names: cap at first 20
    if total_cols > 20:
        col_display = ", ".join(header[:20]) + f", ... ({total_cols - 20} more)"
    else:
        col_display = ", ".join(header)

    # Format sample rows: show at most 8 columns per row for readability
    max_display_cols = 8
    sample_lines: list[str] = []
    display_header = header[:max_display_cols]
    if total_cols > max_display_cols:
        display_header.append(f"... ({total_cols - max_display_cols} more)")
    sample_lines.append("  " + " | ".join(display_header))
    sample_lines.append("  " + "-" * min(80, len(sample_lines[0])))
    for row in sample_rows:
        display_row = row[:max_display_cols]
        if total_cols > max_display_cols:
            display_row.append("...")
        sample_lines.append("  " + " | ".join(display_row))

    # Pandas read hint
    if suffix == ".tsv":
        read_hint = f"df = pd.read_csv('{relative_path}', sep='\\t')"
    else:
        read_hint = f"df = pd.read_csv('{relative_path}')"

    return (
        f"[Data file preview: {relative_path}]\n"
        f"Size: {size_mb:.1f} MB | Rows: {total_rows} | Columns: {total_cols}\n"
        f"\n"
        f"Column names (first 20 of {total_cols}):\n"
        f"  {col_display}\n"
        f"\n"
        f"First 5 rows:\n"
        + "\n".join(sample_lines)
        + f"\n\nTo analyze this file, use run_code with pandas:\n"
        f"  {read_hint}"
    )


def _build_text_preview(
    file_path: Path, relative_path: str, file_size: int, *, max_chars: int = 4000
) -> str:
    """Return a head preview of a large text file, capped by both lines and chars.

    For non-tabular text files that exceed the size threshold, returning the
    full content would waste context. Capping by chars (not just lines) handles
    wide-column CSVs and other files where individual lines can be huge.
    """
    collected: list[str] = []
    char_count = 0
    lines_read = 0
    with open(file_path, encoding="utf-8") as fh:
        for line in fh:
            if lines_read >= 100 or char_count + len(line) > max_chars:
                break
            collected.append(line)
            char_count += len(line)
            lines_read += 1

    # Count total lines via raw scan
    with open(file_path, "rb") as fh:
        total_lines = sum(1 for _ in fh)

    size_mb = file_size / (1024 * 1024)
    preview_text = "".join(collected)

    return (
        f"[Large text file: {relative_path}]\n"
        f"Size: {size_mb:.1f} MB | Total lines: {total_lines}\n"
        f"Showing first {lines_read} lines:\n"
        f"\n"
        f"{preview_text}\n"
        f"\n"
        f"[Truncated — use run_code to process the full file programmatically.]"
    )


class WorkspaceError(Exception):
    """Raised when a workspace operation's precondition is not met.

    Tool implementations catch this and convert it to a ToolError,
    which the registry formats as an error tool result for the LLM.
    """


def resolve_path_within_root(root: Path, relative_path: str) -> Path:
    """Resolve a workspace-relative path and reject escapes outside root."""

    target_path = (root / relative_path).resolve()
    root_path = root.resolve()
    if not target_path.is_relative_to(root_path):
        raise WorkspaceError(f"Path escapes workspace root: {relative_path}")
    return target_path


class Workspace:
    """Manages the workspace filesystem — the project's system of record.

    Design invariant: every write method atomically updates project_state.json
    alongside the file operation, so the two never diverge.
    """

    def __init__(self, root: Path) -> None:
        """Bind to a workspace root directory. Does NOT initialize — call initialize() explicitly."""
        self.root = root

    # ── Initialization ──

    def initialize(self, data_catalog_path: Path | None = None) -> None:
        """Create directory structure and initial project_state.json.

        If data_catalog_path is provided, copies it to workspace/data_catalog.json.
        Idempotent: safe to call on an already-initialized workspace.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "plans").mkdir(parents=True, exist_ok=True)
        (self.root / "stages").mkdir(parents=True, exist_ok=True)

        state_path = self.root / "project_state.json"
        if not state_path.exists():
            self._save_state(ProjectState())

        if data_catalog_path is not None:
            shutil.copyfile(data_catalog_path, self.root / "data_catalog.json")

    # ── Read operations ──

    def get_state(self) -> ProjectState:
        """Read and return the current ProjectState from project_state.json."""
        state = self._load_state()
        for stage_id in state.stages:
            state.stages[stage_id].stage_status = self._derive_stage_status(state, stage_id)
        return state

    def _require_stage(self, state: ProjectState, stage_id: str) -> StageState:
        """Return the existing stage state or raise a workspace error."""

        stage = state.stages.get(stage_id)
        if stage is None:
            raise WorkspaceError(f"Stage not found: {stage_id}")
        return stage

    def _load_state(self) -> ProjectState:
        """Read the persisted ProjectState from disk without deriving stage statuses."""
        state_path = self.root / "project_state.json"
        return ProjectState.model_validate_json(state_path.read_text(encoding="utf-8"))

    def _derive_stage_status(self, state: ProjectState, stage_id: str) -> StageStatus:
        """Derive a stage's lifecycle status from an already-loaded ProjectState."""
        stage = state.stages.get(stage_id)
        if stage is None:
            raise WorkspaceError(f"Stage not found: {stage_id}")

        if stage.plan_status is None:
            return "planning"

        if stage.plan_status == "drafting":
            return "planning"

        if stage.plan_status == "approved" and not stage.has_conclusion:
            return "executing"

        if stage.plan_status == "approved" and stage.has_conclusion:
            for task_id in stage.task_ids:
                result_path = self.root / "stages" / stage_id / "tasks" / f"{task_id}_result.json"
                if not result_path.is_file():
                    return "executing"
            return "completed"

        return "executing"

    def read_file(self, relative_path: str) -> str:
        """Read any file in the workspace by relative path.

        Smart dispatch: tabular data files above the size threshold get a
        structured preview (shape, columns, dtypes, sample rows) instead of
        raw content. This prevents large data files from blowing up the LLM
        context while giving agents enough information to write correct
        analysis code.

        Raises:
            WorkspaceError: Path traversal (escapes workspace root) or file not found.

        Returns:
            File content as string, structured preview, or placeholder.
        """
        target_path = resolve_path_within_root(self.root, relative_path)
        if not target_path.is_file():
            raise WorkspaceError(f"File not found: {relative_path}")

        file_size = target_path.stat().st_size

        # Tabular files above threshold get a structured preview.
        # Check this BEFORE the text/binary probe — pandas handles encoding
        # better than raw open() for data files.
        if target_path.suffix.lower() in _TABULAR_EXTENSIONS and file_size > _LARGE_FILE_THRESHOLD:
            return _build_tabular_preview(target_path, relative_path)

        # Try to read as text; binary files get a placeholder
        try:
            content = target_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return _build_binary_placeholder(target_path, relative_path, file_size)

        # Small text files: return in full
        if file_size <= _LARGE_FILE_THRESHOLD:
            return content

        # Large text files: return a head preview
        return _build_text_preview(target_path, relative_path, file_size)

    # ── Write operations (each atomically updates project_state.json) ──

    def write_plan(self, stage_id: str, content: str) -> None:
        """Write a plan draft to plans/{stage_id}_plan.md.

        Side effects on project_state:
        - Creates StageState if stage is new; sets it as current_stage_id.
        - Sets plan_status='drafting', stage_status='planning'.
        - If plan already exists and is 'drafting', overwrites content (replan).

        Raises:
            WorkspaceError: plan_status is 'approved' (cannot overwrite approved plan).
        """
        state = self.get_state()
        stage = state.stages.get(stage_id)
        if stage is not None and stage.plan_status == "approved":
            raise WorkspaceError(f"Plan already approved for stage: {stage_id}")

        plan_path = self.root / "plans" / f"{stage_id}_plan.md"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(content, encoding="utf-8")

        if stage is None:
            stage = StageState(stage_id=stage_id)
            state.stages[stage_id] = stage
        stage.plan_status = "drafting"
        stage.plan_reviewed = False
        stage.has_conclusion = False
        state.current_stage_id = stage_id
        self._save_state(state)

    def mark_plan_reviewed(self, stage_id: str) -> None:
        """Mark a drafted plan as having gone through a user-review round.

        This state is a harness-level approval gate: a plan cannot be approved
        until the main agent has actually routed it through escalate_to_user.
        Replanning resets the flag and requires another review round.
        """

        state = self.get_state()
        stage = self._require_stage(state, stage_id)
        if stage.plan_status == "drafting":
            stage.plan_reviewed = True
            self._save_state(state)

    def approve_plan(self, stage_id: str) -> None:
        """Mark a stage's plan as approved, transitioning to execution.

        Preconditions:
        - Plan file exists for this stage_id.
        - plan_status == 'drafting'.
        - plan_reviewed == True (the plan has already gone through a user review round).

        Side effects: sets plan_status='approved', stage_status='executing'.

        Raises:
            WorkspaceError: Preconditions not met.
        """
        state = self.get_state()
        plan_path = self.root / "plans" / f"{stage_id}_plan.md"
        if not plan_path.is_file():
            raise WorkspaceError(f"Plan file not found for stage: {stage_id}")

        stage = state.stages.get(stage_id)
        if stage is None or stage.plan_status != "drafting":
            raise WorkspaceError(f"Plan is not in drafting status for stage: {stage_id}")
        if not stage.plan_reviewed:
            raise WorkspaceError(
                f"Plan must be reviewed with the user before approval: {stage_id}"
            )

        stage.plan_status = "approved"
        self._save_state(state)

    def register_task(self, stage_id: str, task_id: str) -> None:
        """Register a subagent task ID in the project state.

        Precondition: stage's plan_status == 'approved'.
        Side effect: appends task_id to stage's task_ids list if it is not already registered.

        Raises:
            WorkspaceError: Stage plan not approved.
        """
        state = self.get_state()
        stage = self._require_stage(state, stage_id)
        if stage.plan_status != "approved":
            raise WorkspaceError(f"Stage plan is not approved: {stage_id}")

        if task_id not in stage.task_ids:
            stage.task_ids.append(task_id)
        self._save_state(state)

    def write_task_result(self, stage_id: str, task_id: str, result: TaskResult) -> None:
        """Write subagent result to stages/{stage_id}/tasks/{task_id}_result.json.

        Preconditions:
        - stage_id exists in project state.
        - task_id is already registered on that stage.

        Serializes the TaskResult model to JSON internally.
        """
        state = self.get_state()
        stage = self._require_stage(state, stage_id)
        if task_id not in stage.task_ids:
            raise WorkspaceError(
                f"Task is not registered for stage {stage_id}: {task_id}"
            )
        result_path = self.root / "stages" / stage_id / "tasks" / f"{task_id}_result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(result.model_dump_json(), encoding="utf-8")
        self._save_state(state)

    def write_artifact(self, stage_id: str, filename: str, content: bytes | str) -> str:
        """Write an artifact to stages/{stage_id}/outputs/{filename}.

        Precondition: stage_id exists in project state.

        Returns:
            Workspace-relative path to the written file.
        """
        state = self.get_state()
        self._require_stage(state, stage_id)
        outputs_dir = self.root / "stages" / stage_id / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = (outputs_dir / filename).resolve()
        if not artifact_path.is_relative_to(outputs_dir.resolve()):
            raise WorkspaceError(f"Artifact path escapes stage outputs: {filename}")

        if isinstance(content, bytes):
            artifact_path.write_bytes(content)
        else:
            artifact_path.write_text(content, encoding="utf-8")

        self._save_state(state)
        return str(artifact_path.relative_to(self.root))

    def write_conclusion(self, stage_id: str, content: str) -> None:
        """Write stage conclusion to stages/{stage_id}/conclusion.md.

        Precondition: stage_status == 'executing'.
        Side effect: sets has_conclusion=True.

        Raises:
            WorkspaceError: Stage is not in 'executing' status.
        """
        state = self.get_state()
        stage = self._require_stage(state, stage_id)
        if stage.stage_status != "executing":
            raise WorkspaceError(f"Stage is not executing: {stage_id}")

        conclusion_path = self.root / "stages" / stage_id / "conclusion.md"
        conclusion_path.parent.mkdir(parents=True, exist_ok=True)
        conclusion_path.write_text(content, encoding="utf-8")
        stage.has_conclusion = True
        self._save_state(state)

    # ── Internal helpers (not part of public interface) ──

    def _save_state(self, state: ProjectState) -> None:
        """Atomically write project_state.json."""
        state.updated_at = datetime.now(tz=timezone.utc)
        state_path = self.root / "project_state.json"
        tmp_path = self.root / "project_state.json.tmp"
        tmp_path.write_text(state.model_dump_json(), encoding="utf-8")
        tmp_path.replace(state_path)
