"""Population creation, evaluation, and survivor selection."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.tours import initial_population, population_lengths


IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class PopulationState:
    """A population and the evaluated length of every individual."""

    tours: IntArray
    lengths: IntArray

    def __post_init__(self) -> None:
        if self.tours.ndim != 2:
            raise ValueError("Population tours must be two-dimensional.")

        if self.lengths.ndim != 1:
            raise ValueError("Population lengths must be one-dimensional.")

        if len(self.tours) != len(self.lengths):
            raise ValueError(
                "Each tour must have one corresponding length."
            )

        if len(self.tours) == 0:
            raise ValueError("Population cannot be empty.")

    @property
    def best_index(self) -> int:
        return int(np.argmin(self.lengths))

    @property
    def best_tour(self) -> IntArray:
        return self.tours[self.best_index]

    @property
    def best_length(self) -> int:
        return int(self.lengths[self.best_index])

    @property
    def mean_length(self) -> float:
        return float(self.lengths.mean())


def create_population(
    population_size: int,
    city_count: int,
    distances: IntArray,
    rng: np.random.Generator,
) -> PopulationState:
    """Create and evaluate the initial random population."""
    tours = initial_population(
        size=population_size,
        city_count=city_count,
        rng=rng,
    )

    return evaluate_population(
        tours=tours,
        distances=distances,
    )


def evaluate_population(
    tours: IntArray,
    distances: IntArray,
) -> PopulationState:
    """Calculate the tour length of every individual."""
    if tours.ndim != 2:
        raise ValueError("Tours must be a two-dimensional array.")

    lengths = population_lengths(
        population=tours,
        distances=distances,
    ).astype(np.int64, copy=False)

    return PopulationState(
        tours=tours,
        lengths=lengths,
    )


def replace_worst(
    population: PopulationState,
    tour: IntArray,
    tour_length: int,
) -> PopulationState:
    """Replace the population's worst individual with a supplied tour."""
    worst_index = int(np.argmax(population.lengths))

    updated_tours = population.tours.copy()
    updated_lengths = population.lengths.copy()

    updated_tours[worst_index] = tour
    updated_lengths[worst_index] = tour_length

    return PopulationState(
        tours=updated_tours,
        lengths=updated_lengths,
    )


def select_survivors(
    current: PopulationState,
    children: PopulationState,
    population_size: int,
    elite_count: int,
) -> PopulationState:
    """
    Preserve elite parents and fill remaining places with best children.
    """
    if population_size != len(current.tours):
        raise ValueError(
            "Population size must match the current population."
        )

    if not 0 <= elite_count < population_size:
        raise ValueError("Elite count is outside the valid range.")

    elite_indices = np.argsort(current.lengths)[:elite_count]

    child_count = population_size - elite_count
    child_indices = np.argsort(children.lengths)[:child_count]

    next_tours = np.concatenate(
        (
            current.tours[elite_indices],
            children.tours[child_indices],
        ),
        axis=0,
    )

    next_lengths = np.concatenate(
        (
            current.lengths[elite_indices],
            children.lengths[child_indices],
        ),
        axis=0,
    )

    return PopulationState(
        tours=next_tours,
        lengths=next_lengths,
    )