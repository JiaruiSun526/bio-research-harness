"""Domain models for the research agent system.

This module defines all data contracts shared across modules:
- Project state (persisted to workspace/project_state.json)
- Task specifications and results (subagent interface)
- LLM interaction types (isolating litellm dependency)
- Loop and run result types
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Return the current time in UTC as a timezone-aware datetime."""

    return datetime.now(tz=timezone.utc)

# ── Status type aliases ──

PlanStatus = Literal["drafting", "approved"]
"""Plan lifecycle: drafting (editable) → approved (locked for execution)."""

StageStatus = Literal["planning", "executing", "completed"]
"""Stage lifecycle: planning (forming plan) → executing (subagents working) → completed."""

StopReason = Literal["completed", "max_turns_reached", "error", "in_progress"]
"""Why the agentic loop exited. 'in_progress' is used only in checkpoint saves."""

TaskStatus = Literal["success", "failure", "partial"]
"""Outcome of a subagent task execution."""

SubAgentRole = Literal["general", "data_analyst", "visualization"]
"""Supported sub-agent roles exposed to the main agent."""


# ── Project state (persisted to workspace/project_state.json) ──


class StageState(BaseModel):
    """State of a single research stage within the project.

    Each stage goes through: planning → executing → completed.
    A stage has at most one plan, zero or more tasks, and at most one conclusion.
    """

    stage_id: str
    plan_status: PlanStatus | None = None
    plan_reviewed: bool = False
    stage_status: StageStatus = Field(default="planning", exclude=True)
    task_ids: list[str] = Field(default_factory=list)
    has_conclusion: bool = False


class ProjectState(BaseModel):
    """Persistent project state — the workspace's system of record.

    Harness maintains this file; every tool-triggered state change
    atomically updates it. Main agent receives a summary of this
    in each turn's system message, so core state survives even if
    conversation history is truncated by the LLM context window.
    """

    current_stage_id: str | None = None
    stages: dict[str, StageState] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


# ── Subagent task interface ──


class TaskSpec(BaseModel):
    """Full specification for a subagent task, passed to dispatch_subagent.

    Main agent constructs this when dispatching work. The harness uses it
    to set up the subagent's isolated context (messages, tools, prompt).
    """

    task_id: str
    stage_id: str
    task_description: str
    role: SubAgentRole = "general"
    system_prompt: str | None = None  # optional override — bypasses role-based template
    max_turns: int = 30


class TaskResult(BaseModel):
    """Structured result returned by subagent to main agent.

    Main agent only sees this summary — full execution output
    (code, stdout, intermediate files) stays in workspace files.
    This information boundary is enforced by the harness, not by
    the main agent's self-discipline.
    """

    task_id: str
    status: TaskStatus
    summary: str
    artifact_paths: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    error: str | None = None
    artifact_descriptions: dict[str, str] = Field(default_factory=dict)


class SubAgentCompletion(BaseModel):
    """Structured completion payload submitted by a subagent terminal tool."""

    summary: str
    blockers: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    artifact_descriptions: dict[str, str] = Field(default_factory=dict)
    """Map of filename → one-line description for each output file produced."""


# ── Escalation / review context ──


class ReviewContext(BaseModel):
    """Context provided to user during escalation or review.

    Contains workspace-relative paths to artifacts the user can inspect.
    For SimulatedUser, these file contents are loaded into the current
    LLM call but NOT retained across turns.
    """

    stage_id: str | None = None
    artifact_paths: list[str] = Field(default_factory=list)


# ── LLM interaction types (isolating litellm dependency) ──


class ToolCallRequest(BaseModel):
    """A single tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


class LLMResponse(BaseModel):
    """Normalized response from any LLM provider.

    llm_client.py converts litellm's raw response into this format.
    No other module imports litellm types directly.

    Cache token fields (cache_creation_tokens, cache_read_tokens) are populated
    when the provider returns prompt-cache usage (Anthropic explicit cache,
    OpenAI auto cache). Both are 0 when caching is off or unsupported.
    """

    content: str | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


# ── Loop and run results ──


class AgentLoopResult(BaseModel):
    """Result of a single agentic loop execution (main or sub agent)."""

    stop_reason: StopReason
    final_response: str | None = None
    error_message: str | None = None
    tool_call_count: int = 0
    turn_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


class RunResult(BaseModel):
    """Result of a complete research harness run."""

    stop_reason: StopReason
    stages_completed: int = 0
    tool_call_count: int = 0
    turn_count: int = 0
    escalation_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    error_message: str | None = None
    final_response: str | None = None
