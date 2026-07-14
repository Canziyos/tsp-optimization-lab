"""Production of one complete offspring population."""

import numpy as np
from numpy.typing import NDArray

from .config import GAConfig
from .crossover import order_crossover
from .mutation import mutate_inversion
from .population import PopulationState
from .selection import tournament_select


IntArray = NDArray[np.int64]


def create_offspring(
    population: PopulationState,
    config: GAConfig,
    rng: np.random.Generator,
) -> IntArray:
    """Create one full offspring population."""
    children: list[IntArray] = []

    while len(children) < config.population_size:
        first_parent = tournament_select(
            population=population.tours,
            lengths=population.lengths,
            tournament_size=config.tournament_size,
            win_probability=config.tournament_win_probability,
            rng=rng,
        )

        second_parent = tournament_select(
            population=population.tours,
            lengths=population.lengths,
            tournament_size=config.tournament_size,
            win_probability=config.tournament_win_probability,
            rng=rng,
        )

        first_child, second_child = _reproduce_pair(
            first_parent=first_parent,
            second_parent=second_parent,
            config=config,
            rng=rng,
        )

        children.append(first_child)

        if len(children) < config.population_size:
            children.append(second_child)

    return np.stack(children)


def _reproduce_pair(
    first_parent: IntArray,
    second_parent: IntArray,
    config: GAConfig,
    rng: np.random.Generator,
) -> tuple[IntArray, IntArray]:
    """Create and mutate two children from two parents."""
    if rng.random() < config.crossover_probability:
        first_child = order_crossover(
            first_parent=first_parent,
            second_parent=second_parent,
            rng=rng,
        )

        second_child = order_crossover(
            first_parent=second_parent,
            second_parent=first_parent,
            rng=rng,
        )
    else:
        first_child = first_parent.copy()
        second_child = second_parent.copy()

    first_child = mutate_inversion(
        tour=first_child,
        probability=config.mutation_probability,
        rng=rng,
    )

    second_child = mutate_inversion(
        tour=second_child,
        probability=config.mutation_probability,
        rng=rng,
    )

    return first_child, second_child