"""Tour construction behaviour for one ant."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import ACOConfig
from .pheromone import PheromoneMap


IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(slots=True)
class Ant:
    """An ant that constructs a closed tour through all cities."""

    city_count: int
    start_city: int = 0

    def __post_init__(self) -> None:
        if self.city_count < 3:
            raise ValueError("At least three cities are required.")

        if not 0 <= self.start_city < self.city_count:
            raise ValueError("Start city is outside the valid city range.")

    def construct_tour(
        self,
        pheromone: PheromoneMap,
        heuristic: FloatArray,
        config: ACOConfig,
        rng: np.random.Generator,
    ) -> IntArray:
        """Construct one complete closed tour."""
        self._validate_inputs(pheromone, heuristic)

        tour = np.empty(self.city_count + 1, dtype=np.int64)
        tour[0] = self.start_city
        tour[-1] = self.start_city

        unvisited = np.ones(self.city_count, dtype=np.bool_)
        unvisited[self.start_city] = False

        current_city = self.start_city

        for position in range(1, self.city_count):
            candidates = self._candidate_cities(unvisited)

            weights = self._transition_weights(
                current_city=current_city,
                candidates=candidates,
                pheromone=pheromone,
                heuristic=heuristic,
                alpha=config.alpha,
                beta=config.beta,
            )

            next_city = self._choose_next_city(
                candidates=candidates,
                weights=weights,
                rng=rng,
            )

            tour[position] = next_city
            unvisited[next_city] = False
            current_city = next_city

        return tour

    @staticmethod
    def _candidate_cities(unvisited: BoolArray) -> IntArray:
        """Return the indices of all cities that have not been visited."""
        return np.flatnonzero(unvisited).astype(np.int64, copy=False)

    @staticmethod
    def _transition_weights(
        current_city: int,
        candidates: IntArray,
        pheromone: PheromoneMap,
        heuristic: FloatArray,
        alpha: float,
        beta: float,
    ) -> FloatArray:
        """Calculate the relative desirability of each candidate city."""
        pheromone_strength = pheromone.values[current_city, candidates]
        heuristic_strength = heuristic[current_city, candidates]

        weights = (
            np.power(pheromone_strength, alpha)
            * np.power(heuristic_strength, beta)
        )

        return np.asarray(weights, dtype=np.float64)

    @staticmethod
    def _choose_next_city(
        candidates: IntArray,
        weights: FloatArray,
        rng: np.random.Generator,
    ) -> int:
        """Randomly choose a candidate according to normalized weights."""
        if len(candidates) == 0:
            raise ValueError("No candidate cities are available.")

        if len(candidates) != len(weights):
            raise ValueError("Candidates and weights must have equal lengths.")

        if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
            raise ValueError("Transition weights must be finite and non-negative.")

        total_weight = float(weights.sum())

        if total_weight <= 0.0:
            # No candidate is preferred, so choose uniformly.
            return int(rng.choice(candidates))

        probabilities = weights / total_weight

        return int(rng.choice(candidates, p=probabilities))

    def _validate_inputs(
        self,
        pheromone: PheromoneMap,
        heuristic: FloatArray,
    ) -> None:
        """Check that the matrices match the ant's city count."""
        expected_shape = (self.city_count, self.city_count)

        if pheromone.values.shape != expected_shape:
            raise ValueError(
                f"Pheromone matrix must have shape {expected_shape}."
            )

        if heuristic.shape != expected_shape:
            raise ValueError(
                f"Heuristic matrix must have shape {expected_shape}."
            )

        if np.any(heuristic < 0.0):
            raise ValueError("Heuristic values cannot be negative.")

        if not np.all(np.isfinite(heuristic)):
            raise ValueError("Heuristic matrix contains non-finite values.")