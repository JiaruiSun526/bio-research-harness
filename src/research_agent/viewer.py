"""Streamlit session viewer — human-friendly visualization of run data.

Reads session.json produced by the harness and renders an interactive
conversation timeline with sidebar summary and filters.

Run:
    streamlit run src/research_agent/viewer.py -- <path-to-session.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st


# ── Data loading ──


def load_session(path: str) -> dict:
    """Load and return session.json data."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── Sidebar ──


def render_sidebar(session: dict) -> dict[str, bool]:
    """Render sidebar with run summary, stage progression, and filters.

    Returns a dict of active filter flags.
    """
    st.sidebar.title("Run Summary")

    run_result = session.get("run_result", {})
    st.sidebar.markdown(f"**Model:** `{session.get('model', 'unknown')}`")
    st.sidebar.markdown(f"**Stop reason:** {run_result.get('stop_reason', 'unknown')}")
    st.sidebar.markdown(f"**Stages completed:** {run_result.get('stages_completed', 0)}")
    st.sidebar.markdown(f"**Tool calls:** {run_result.get('tool_call_count', 0)}")
    st.sidebar.markdown(f"**Escalations:** {run_result.get('escalation_count', 0)}")

    prompt_tokens = run_result.get("prompt_tokens", 0)
    completion_tokens = run_result.get("completion_tokens", 0)
    st.sidebar.markdown(f"**Tokens:** {prompt_tokens:,} prompt + {completion_tokens:,} completion")

    st.sidebar.markdown(f"**Started:** {session.get('started_at', '')}")
    st.sidebar.markdown(f"**Completed:** {session.get('completed_at', '')}")

    # Stage progression
    st.sidebar.markdown("---")
    st.sidebar.subheader("Stage Progression")
    _render_stage_progression(session)

    # Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    show_system = st.sidebar.checkbox("System messages", value=False)
    show_main = st.sidebar.checkbox("Main agent", value=True)
    show_sub = st.sidebar.checkbox("Sub agents", value=True)
    show_tools = st.sidebar.checkbox("Tool calls", value=True)

    return {
        "show_system": show_system,
        "show_main": show_main,
        "show_sub": show_sub,
        "show_tools": show_tools,
    }


def _render_stage_progression(session: dict) -> None:
    """Show per-stage status from the recorded project-state snapshot."""

    project_state = session.get("project_state", {})
    stages = project_state.get("stages", {})
    for stage_id in sorted(stages):
        stage = stages.get(stage_id, {})
        icon = stage.get("stage_status", "unknown")
        task_ids = stage.get("task_ids", [])
        st.sidebar.markdown(f"- **{stage_id}** ({icon}) — {len(task_ids)} task(s)")


# ── Message rendering ──


def render_conversation(session: dict, filters: dict[str, bool]) -> None:
    """Render the main conversation timeline with subagent inlining."""
    st.header("Conversation Timeline")

    messages = session.get("main_conversation", [])
    subagent_runs = session.get("subagent_runs", [])

    # Index subagent runs by task_id for inline rendering
    sub_by_task: dict[str, dict] = {}
    for run in subagent_runs:
        sub_by_task[run.get("task_id", "")] = run

    # Build a set of tool_call_ids that correspond to escalate_to_user calls.
    # Their tool results contain user responses and should render as user messages.
    escalation_call_ids: set[str] = set()
    for msg in messages:
        for tc in msg.get("tool_calls", []):
            if tc.get("name") == "escalate_to_user":
                call_id = tc.get("id", "")
                if call_id:
                    escalation_call_ids.add(call_id)

    # Track which dispatch tool calls we've rendered subagent for
    rendered_subs: set[str] = set()

    for msg in messages:
        role = msg.get("role", "unknown")

        if role == "system":
            if filters.get("show_system"):
                _render_system_message(msg)
            continue

        if role == "user":
            if filters.get("show_main"):
                _render_user_message(msg)
            continue

        if role == "assistant":
            if not filters.get("show_main"):
                continue
            _render_assistant_message(msg, filters)
            # Check for dispatch_subagent tool calls to inline subagent
            if filters.get("show_sub"):
                for tc in msg.get("tool_calls", []):
                    if tc.get("name") == "dispatch_subagent":
                        task_id = tc.get("arguments", {}).get("task_id", "")
                        if task_id and task_id in sub_by_task and task_id not in rendered_subs:
                            rendered_subs.add(task_id)
                            _render_subagent_block(sub_by_task[task_id], filters)
            continue

        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            if tool_call_id in escalation_call_ids:
                # This is a user response via escalate_to_user — render as user
                _render_user_message(msg)
            elif filters.get("show_tools"):
                _render_tool_result(msg)
            continue


def _render_system_message(msg: dict) -> None:
    """Render system message — collapsed by default."""
    content = msg.get("content", "")
    with st.expander("System message", expanded=False):
        st.code(content[:2000] + ("..." if len(content) > 2000 else ""), language=None)


def _render_user_message(msg: dict) -> None:
    """Render user / initial goal message."""
    content = msg.get("content", "")
    st.chat_message("user").markdown(content)


def _render_assistant_message(msg: dict, filters: dict[str, bool]) -> None:
    """Render assistant text and tool calls."""
    content = msg.get("content")
    tool_calls = msg.get("tool_calls", [])

    if content:
        st.chat_message("assistant").markdown(content)

    if filters.get("show_tools"):
        for tc in tool_calls:
            name = tc.get("name", "unknown")
            args = tc.get("arguments", {})
            args_preview = _format_args_preview(args)
            icon = _get_tool_icon(name)
            st.markdown(
                f"<div style='background:#f0f0f0;padding:4px 10px;border-radius:4px;"
                f"margin:2px 0;font-family:monospace;font-size:0.85em'>"
                f"{icon} <b>{name}</b>({args_preview})</div>",
                unsafe_allow_html=True,
            )


def _render_tool_result(msg: dict) -> None:
    """Render a tool result message."""
    content = msg.get("content", "")
    truncated = content[:500]
    if len(content) > 500:
        truncated += "..."
    st.markdown(
        f"<div style='background:#f8f8f8;padding:4px 10px 4px 20px;border-left:2px solid #ccc;"
        f"margin:2px 0 6px 0;font-family:monospace;font-size:0.8em;color:#555'>"
        f"&rarr; {truncated}</div>",
        unsafe_allow_html=True,
    )


def _render_subagent_block(run: dict, filters: dict[str, bool]) -> None:
    """Render a subagent run as a bordered block."""
    task_id = run.get("task_id", "unknown")
    role = run.get("role", "general")
    task_result = run.get("task_result", {})
    status = task_result.get("status", "unknown")

    status_icon = {"success": "ok", "failure": "FAIL", "partial": "partial"}.get(status, "?")

    with st.expander(f"Sub Agent: {task_id} (role={role}, status={status_icon})", expanded=False):
        st.markdown(f"**Task:** {run.get('task_description', '')}")
        st.markdown(f"**Result:** {task_result.get('summary', '')}")

        if task_result.get("artifact_paths"):
            st.markdown(f"**Artifacts:** {', '.join(task_result['artifact_paths'])}")

        # Render subagent conversation
        conversation = run.get("conversation", [])
        for msg in conversation:
            msg_role = msg.get("role", "unknown")
            if msg_role == "system" and not filters.get("show_system"):
                continue
            if msg_role == "user":
                st.markdown(f"> **User:** {msg.get('content', '')}")
            elif msg_role == "assistant":
                content = msg.get("content")
                if content:
                    st.markdown(f"> **Agent:** {content}")
                for tc in msg.get("tool_calls", []):
                    name = tc.get("name", "unknown")
                    args_preview = _format_args_preview(tc.get("arguments", {}))
                    st.markdown(f"> `{name}({args_preview})`")
            elif msg_role == "tool" and filters.get("show_tools"):
                tool_content = msg.get("content", "")
                truncated = tool_content[:200] + ("..." if len(tool_content) > 200 else "")
                st.markdown(f"> → _{truncated}_")


# ── Helpers ──


def _format_args_preview(args: object) -> str:
    """Format tool call arguments as a compact preview string."""

    if not isinstance(args, dict):
        preview = json.dumps(args)
        if len(preview) > 120:
            preview = preview[:117] + "..."
        return preview

    parts: list[str] = []
    for key, value in args.items():
        if isinstance(value, str) and len(value) > 60:
            value = value[:57] + "..."
        parts.append(f"{key}={json.dumps(value)}")
    preview = ", ".join(parts)
    if len(preview) > 120:
        preview = preview[:117] + "..."
    return preview


def _get_tool_icon(name: str) -> str:
    """Return a text marker for a tool name."""
    icons = {
        "save_plan": "[plan]",
        "approve_plan": "[approve]",
        "dispatch_subagent": "[dispatch]",
        "save_conclusion": "[conclude]",
        "escalate_to_user": "[escalate]",
        "finish_run": "[finish]",
        "read_file": "[read]",
        "write_file": "[write]",
        "run_code": "[code]",
    }
    return icons.get(name, "[tool]")


# ── Main ──


def _discover_runs(runs_dir: Path) -> list[tuple[str, Path]]:
    """Find all session.json files under runs/ and return (label, path) pairs."""
    results: list[tuple[str, Path]] = []
    if not runs_dir.is_dir():
        return results
    for workspace in sorted(runs_dir.iterdir(), reverse=True):
        session_file = workspace / "session.json"
        if session_file.is_file():
            # Build a label: folder name + stop reason + stages
            try:
                data = json.loads(session_file.read_text(encoding="utf-8"))
                rr = data.get("run_result", {})
                model = data.get("model", "?")
                stop = rr.get("stop_reason", "?")
                stages = rr.get("stages_completed", "?")
                label = f"{workspace.name}  [{model}]  {stop}, {stages} stages"
            except Exception:
                label = workspace.name
            results.append((label, session_file))
    return results


def main() -> None:
    """Streamlit app entry point."""
    st.set_page_config(page_title="Research Agent Session Viewer", layout="wide")
    st.title("Research Agent Session Viewer")

    # Determine project root (viewer lives in src/research_agent/)
    project_root = Path(__file__).resolve().parents[2]
    runs_dir = project_root / "runs"

    # Always discover available runs for sidebar selection
    available_runs = _discover_runs(runs_dir)

    # Command-line argument sets the default selection
    cli_path: str | None = None
    if len(sys.argv) > 1:
        cli_path = sys.argv[1]

    session_path: str | None = None

    if available_runs:
        st.sidebar.title("Select Run")
        labels = [label for label, _ in available_runs]
        paths = [str(p) for _, p in available_runs]

        # Find default index: match CLI argument if provided
        default_idx = 0
        if cli_path:
            cli_resolved = str(Path(cli_path).resolve())
            for i, p in enumerate(paths):
                if str(Path(p).resolve()) == cli_resolved:
                    default_idx = i
                    break

        selected_idx = st.sidebar.selectbox(
            "Available runs",
            range(len(labels)),
            index=default_idx,
            format_func=lambda i: labels[i],
        )
        session_path = paths[selected_idx]
    elif cli_path:
        session_path = cli_path
    else:
        uploaded = st.file_uploader("Upload session.json", type=["json"])
        if uploaded is None:
            st.info("No runs found. Upload a session.json or pass one as argument.")
            return
        session = json.loads(uploaded.read())
        filters = render_sidebar(session)
        render_conversation(session, filters)
        return

    if session_path is None or not Path(session_path).is_file():
        st.error(f"Session file not found: {session_path}")
        return
    session = load_session(session_path)

    filters = render_sidebar(session)
    render_conversation(session, filters)


if __name__ == "__main__":
    main()
