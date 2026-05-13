"""UserAgent protocol — the interface between main agent and user.

The main agent interacts with the user exclusively through escalate_to_user,
which routes to UserAgent.respond(). The main agent does not know whether
the other side is a real human or an LLM-simulated user.
"""

from __future__ import annotations

from typing import Protocol

from ..schemas import ReviewContext


class UserAgent(Protocol):
    """Interface for user interaction — real human or LLM-simulated.

    This is the sole communication channel between the main agent and
    the user. All project-level decisions, plan reviews, and escalations
    flow through this interface.
    """

    def respond(self, message: str, context: ReviewContext | None = None) -> str:
        """Receive a message from the main agent and return the user's response.

        Args:
            message: Natural language message from the main agent
                (plan proposal, status update, escalation question, etc.).
            context: Optional review context with workspace-relative paths
                to artifacts the user can inspect before responding.

        Returns:
            User's natural language response.
        """
        ...
