"""Shared result models for optimization algorithms."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class HistoryPoint:
    """One recorded optimization step."""

    step: int
    best_length: int
    mean_length: float

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("History step cannot be negative.")

        if self.best_length <= 0:
            raise ValueError("Best tour length must be positive.")

        if self.mean_length <= 0.0:
            raise ValueError("Mean tour length must be positive.")


@dataclass(frozen=True, slots=True)
class SolverResult:
    """Final output produced by one optimization algorithm."""

    algorithm: str
    best_tour: IntArray
    best_length: int
    best_step: int
    history: tuple[HistoryPoint, ...]
    seed: int | None

    def __post_init__(self) -> None:
        if not self.algorithm.strip():
            raise ValueError("Algorithm name cannot be empty.")

        if self.best_tour.ndim != 1:
            raise ValueError("Best tour must be one-dimensional.")

        if self.best_length <= 0:
            raise ValueError("Best tour length must be positive.")

        if self.best_step < 0:
            raise ValueError("Best step cannot be negative.")

        if len(self.history) == 0:
            raise ValueError("Solver history cannot be empty.")

        if self.best_step > self.history[-1].step:
            raise ValueError(
                "Best step cannot occur after the final history step."
            )