"""Mutation operators for closed TSP tours."""

import numpy as np
from numpy.typing import NDArray


IntArray = NDArray[np.int64]


def mutate_inversion(
    tour: IntArray,
    probability: float,
    rng: np.random.Generator,
) -> IntArray:
    """
    Reverse a random interior segment with the given probability.

    The depot at the beginning and end remains unchanged.
    """
    if tour.ndim != 1:
        raise ValueError("Tour must be one-dimensional.")

    if len(tour) < 4:
        raise ValueError(
            "Closed tours must contain at least three cities."
        )

    if tour[0] != 0 or tour[-1] != 0:
        raise ValueError("Tour must start and end at city zero.")

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "Mutation probability must be between zero and one."
        )

    mutated = tour.copy()

    if rng.random() >= probability:
        return mutated

    left, right = sorted(
        rng.choice(
            np.arange(1, len(tour) - 1),
            size=2,
            replace=False,
        )
    )

    mutated[left : right + 1] = (
        mutated[left : right + 1][::-1]
    )

    return mutated