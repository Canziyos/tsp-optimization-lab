"""Crossover operators for closed TSP tours."""

import numpy as np
from numpy.typing import NDArray


IntArray = NDArray[np.int64]


def order_crossover(
    first_parent: IntArray,
    second_parent: IntArray,
    rng: np.random.Generator,
) -> IntArray:
    """
    Create one child using order crossover.

    Tours start and end at depot city zero. Only interior cities
    participate in crossover.
    """
    _validate_parents(first_parent, second_parent)

    interior_count = len(first_parent) - 2

    left, right = sorted(
        rng.choice(
            np.arange(1, len(first_parent) - 1),
            size=2,
            replace=False,
        )
    )

    child = np.full_like(first_parent, fill_value=-1)

    child[0] = 0
    child[-1] = 0

    # Preserve one segment from the first parent.
    child[left : right + 1] = first_parent[left : right + 1]

    preserved_cities = set(
        child[left : right + 1].tolist()
    )

    # Continue reading the second parent after the copied segment,
    # wrapping around the interior of the tour.
    wrapped_positions = [
        1 + ((right + offset - 1) % interior_count)
        for offset in range(1, interior_count + 1)
    ]

    remaining_cities = [
        int(second_parent[position])
        for position in wrapped_positions
        if int(second_parent[position]) not in preserved_cities
    ]

    empty_positions = [
        position
        for position in wrapped_positions
        if child[position] == -1
    ]

    child[empty_positions] = remaining_cities

    return child


def _validate_parents(
    first_parent: IntArray,
    second_parent: IntArray,
) -> None:
    """Validate basic requirements for order crossover."""
    if first_parent.ndim != 1 or second_parent.ndim != 1:
        raise ValueError("Parents must be one-dimensional tours.")

    if first_parent.shape != second_parent.shape:
        raise ValueError("Parents must have equal shapes.")

    if len(first_parent) < 4:
        raise ValueError(
            "Closed tours must contain at least three cities."
        )

    if (
        first_parent[0] != 0
        or first_parent[-1] != 0
        or second_parent[0] != 0
        or second_parent[-1] != 0
    ):
        raise ValueError("Parent tours must start and end at city zero.")

    if not np.array_equal(
        np.sort(first_parent[1:-1]),
        np.sort(second_parent[1:-1]),
    ):
        raise ValueError(
            "Parents must contain the same set of cities."
        )