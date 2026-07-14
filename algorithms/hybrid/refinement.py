"""Local-search refinement used by the hybrid solver."""

import numpy as np
from numpy.typing import NDArray

from algorithms.two_opt import TwoOptConfig, improve_two_opt


IntArray = NDArray[np.int64]


def refine_best_tours(
    tours: IntArray,
    lengths: IntArray,
    distances: IntArray,
    refine_count: int,
    config: TwoOptConfig,
) -> None:
    """
    Improve the best population members using 2-opt.

    Tours and lengths are updated in place.
    """
    _validate_population(
        tours=tours,
        lengths=lengths,
        refine_count=refine_count,
    )

    selected_indices = np.argsort(lengths)[:refine_count]

    for index in selected_indices:
        improved_tour, improved_length, _ = improve_two_opt(
            tour=tours[index],
            distances=distances,
            config=config,
        )

        if improved_length < lengths[index]:
            tours[index] = improved_tour
            lengths[index] = improved_length


def _validate_population(
    tours: IntArray,
    lengths: IntArray,
    refine_count: int,
) -> None:
    """Validate inputs used by population refinement."""
    if tours.ndim != 2:
        raise ValueError("Tours must be a two-dimensional array.")

    if lengths.ndim != 1:
        raise ValueError("Lengths must be one-dimensional.")

    if len(tours) != len(lengths):
        raise ValueError(
            "Each tour must have one corresponding length."
        )

    if len(tours) == 0:
        raise ValueError("Population cannot be empty.")

    if not 1 <= refine_count <= len(tours):
        raise ValueError(
            "Refinement count must be within the population size."
        )