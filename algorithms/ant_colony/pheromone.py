"""Pheromone storage, evaporation, and reinforcement."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class PheromoneMap:
    values: FloatArray

    @classmethod
    def uniform(cls, city_count: int, initial_value: float = 1.0) -> "PheromoneMap":
        if city_count < 2:
            raise ValueError("At least two cities are required.")
        if initial_value <= 0.0:
            raise ValueError("Initial pheromone value must be positive.")
        values = np.full((city_count, city_count), initial_value, dtype=np.float64)
        np.fill_diagonal(values, 0.0)
        return cls(values)

    def evaporate(self, evaporation_rate: float) -> None:
        if not 0.0 <= evaporation_rate <= 1.0:
            raise ValueError("Evaporation rate must be between zero and one.")
        self.values *= 1.0 - evaporation_rate
        np.fill_diagonal(self.values, 0.0)

    def deposit(self, tour: IntArray, amount: float) -> None:
        if tour.ndim != 1 or len(tour) < 4:
            raise ValueError("Tour must be a closed one-dimensional TSP tour.")
        if tour[0] != tour[-1]:
            raise ValueError("Tour must return to its starting city.")
        if amount < 0.0:
            raise ValueError("Deposit amount cannot be negative.")
        if amount == 0.0:
            return
        city_count = self.values.shape[0]
        visited = tour[:-1]
        if len(visited) != city_count or len(np.unique(visited)) != city_count:
            raise ValueError("Tour must visit every city exactly once.")
        if np.any(visited < 0) or np.any(visited >= city_count):
            raise ValueError("Tour contains invalid city indices.")
        first, second = tour[:-1], tour[1:]
        np.add.at(self.values, (first, second), amount)
        np.add.at(self.values, (second, first), amount)

    def deposit_by_length(self, tour: IntArray, tour_length: float, deposit_weight: float) -> None:
        if tour_length <= 0.0 or deposit_weight <= 0.0:
            raise ValueError("Tour length and deposit weight must be positive.")
        self.deposit(tour, deposit_weight / tour_length)

    def deposit_ranked(
        self,
        tours: IntArray,
        lengths: IntArray,
        ranked_ants: int,
        deposit_weight: float,
    ) -> None:
        if tours.ndim != 2 or lengths.ndim != 1 or len(tours) != len(lengths):
            raise ValueError("Tours and lengths must describe the same population.")
        if not 1 <= ranked_ants <= len(tours):
            raise ValueError("Ranked-ant count is outside the valid range.")
        for rank_weight, ant_index in zip(
            range(ranked_ants, 0, -1), np.argsort(lengths)[:ranked_ants]
        ):
            length = float(lengths[ant_index])
            if length <= 0.0:
                raise ValueError("Tour lengths must be positive.")
            self.deposit(tours[ant_index], deposit_weight * rank_weight / length)

    def validate(self) -> None:
        if self.values.ndim != 2 or self.values.shape[0] != self.values.shape[1]:
            raise ValueError("Pheromone matrix must be square.")
        if not np.all(np.isfinite(self.values)) or np.any(self.values < 0.0):
            raise ValueError("Pheromone values must be finite and non-negative.")
        if not np.allclose(np.diag(self.values), 0.0):
            raise ValueError("Pheromone matrix diagonal must be zero.")
        if not np.allclose(self.values, self.values.T):
            raise ValueError("Pheromone matrix must be symmetric.")
