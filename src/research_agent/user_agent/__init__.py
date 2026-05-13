"""UserAgent protocol and implementations (human / simulated)."""

from .base import UserAgent
from .human import HumanUser
from .simulated import SimulatedUser

__all__ = ["UserAgent", "HumanUser", "SimulatedUser"]
