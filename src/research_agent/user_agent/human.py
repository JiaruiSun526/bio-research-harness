"""HumanUser — real user interacting through the terminal."""

from __future__ import annotations

from pathlib import Path

from ..schemas import ReviewContext
from ..workspace import WorkspaceError, resolve_path_within_root


class HumanUser:
    """Real user interacting through terminal stdin/stdout.

    Prints the main agent's message, optionally shows artifact paths
    (converted to absolute paths for easy inspection), and waits for
    the user to type a response.
    """

    def __init__(self, workspace_root: Path) -> None:
        """
        Args:
            workspace_root: Workspace root path, used to resolve
                relative artifact paths to absolute paths for display.
        """
        self.workspace_root: Path = workspace_root

    def respond(self, message: str, context: ReviewContext | None = None) -> str:
        """Print message to terminal, wait for user input, and return it."""
        separator = "─" * 60

        print(separator)
        print(f"Agent: {message}")
        if context and context.artifact_paths:
            print("Review artifacts:")
            for artifact_path in context.artifact_paths:
                try:
                    print(resolve_path_within_root(self.workspace_root, artifact_path))
                except WorkspaceError:
                    print(f"(invalid artifact path: {artifact_path})")
        print(separator)

        return input("You: ")
