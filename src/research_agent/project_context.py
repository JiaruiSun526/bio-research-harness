"""Multi-source project context loader for the stable system block.

Reads project-level markdown rule files (``PROJECT_RULES.md``, ``AGENTS.md``,
``CLAUDE.md``) from one or more search paths and merges them into a single
Markdown block suitable for embedding inside the stable system prompt. The
result is byte-stable across turns by design — the harness loads it once at
construction time and the ContextManager treats it as part of the cache
anchor, so the LLM provider can keep the system prefix cached.

Search semantics:
- Multiple search paths supported, evaluated in the order given. Earlier
  paths render first in the output; if a later file repeats guidance, the
  agent reads top-to-bottom and the last instruction "wins" naturally.
- Within each search path, filenames are tried in a fixed priority order
  (``PROJECT_RULES.md`` > ``AGENTS.md`` > ``CLAUDE.md``). All matches in one
  path are included — they are *not* mutually exclusive.
- Missing files and empty files are silently skipped. An empty result string
  is returned when nothing matched, signaling a no-op to the caller.

This module performs file IO at load time only. The caller is responsible
for caching the returned string for the lifetime of the run.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_PROJECT_CONTEXT_FILENAMES: tuple[str, ...] = (
    "PROJECT_RULES.md",
    "AGENTS.md",
    "CLAUDE.md",
)

_SECTION_HEADER = "## Project Context"


def load_project_context(
    *,
    search_paths: list[Path] | None = None,
    filenames: tuple[str, ...] = DEFAULT_PROJECT_CONTEXT_FILENAMES,
) -> str:
    """Load and merge project context files into a stable Markdown block.

    Args:
        search_paths: Directories to search, in priority order. Earlier
            paths are rendered first. When None or empty, returns "".
        filenames: Filenames to look for in each search path, in priority
            order. Defaults to PROJECT_RULES.md, AGENTS.md, CLAUDE.md.

    Returns:
        A Markdown string starting with "## Project Context" when at least
        one non-empty file was found, otherwise an empty string. The empty
        result lets the caller append unconditionally without producing a
        dangling header.
    """

    paths = list(search_paths or [])
    if not paths:
        return ""

    sections: list[str] = []
    seen_files: set[Path] = set()
    for base in paths:
        for fname in filenames:
            file_path = base / fname
            if not file_path.is_file():
                continue
            resolved = file_path.resolve()
            if resolved in seen_files:
                continue
            seen_files.add(resolved)
            content = file_path.read_text(encoding="utf-8").strip()
            if not content:
                continue
            sections.append(f"### From `{file_path}`\n\n{content}")

    if not sections:
        return ""
    return f"{_SECTION_HEADER}\n\n" + "\n\n".join(sections)
