"""Public interface for ant-colony optimization."""

from .config import ACOConfig
from .solver import solve_ant_colony

__all__ = [
    "ACOConfig",
    "solve_ant_colony",
]