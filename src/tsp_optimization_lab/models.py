"""Shared result models for every optimization method."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class HistoryPoint:
    step: int
    best_length: int
    mean_length: float | None = None


@dataclass(frozen=True, slots=True)
class SolverResult:
    algorithm: str
    best_tour: np.ndarray
    best_length: int
    best_step: int
    history: tuple[HistoryPoint, ...]
    seed: int | None = None


class Solver(Protocol):
    def __call__(self, coordinates: np.ndarray) -> SolverResult: ...

