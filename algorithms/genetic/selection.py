"""Parent-selection operators."""

import numpy as np
from numpy.typing import NDArray


IntArray = NDArray[np.int64]


def tournament_select(
    population: IntArray,
    lengths: IntArray,
    tournament_size: int,
    win_probability: float,
    rng: np.random.Generator,
) -> IntArray:
    """
    Select one parent through probabilistic tournament selection.

    Shorter tour length means better fitness.
    """
    if population.ndim != 2:
        raise ValueError("Population must be a two-dimensional array.")

    if lengths.ndim != 1:
        raise ValueError("Lengths must be one-dimensional.")

    if len(population) != len(lengths):
        raise ValueError(
            "Each population member must have one corresponding length."
        )

    if not 1 <= tournament_size <= len(population):
        raise ValueError(
            "Tournament size must be within the population."
        )

    if not 0.0 <= win_probability <= 1.0:
        raise ValueError(
            "Tournament win probability must be between zero and one."
        )

    candidate_indices = rng.choice(
        len(population),
        size=tournament_size,
        replace=False,
    )

    ranked_indices = candidate_indices[
        np.argsort(lengths[candidate_indices])
    ]

    best_index = int(ranked_indices[0])

    if len(ranked_indices) == 1:
        winner_index = best_index
    elif rng.random() < win_probability:
        winner_index = best_index
    else:
        winner_index = int(rng.choice(ranked_indices[1:]))

    return population[winner_index].copy()