"""Primitive operations used by 2-opt local search."""

import numpy as np
from numpy.typing import NDArray


IntArray = NDArray[np.int64]


def apply_two_opt_move(
    tour: IntArray,
    left: int,
    right: int,
) -> IntArray:
    """
    Reverse one interior segment of a closed tour.

    The first and final depot entries remain unchanged.
    """
    _validate_move(tour, left, right)

    improved = tour.copy()
    improved[left : right + 1] = improved[left : right + 1][::-1]

    return improved


def two_opt_delta(
    tour: IntArray,
    distances: IntArray,
    left: int,
    right: int,
) -> int:
    """
    Return the length change produced by reversing one segment.

    Negative delta means improvement.
    """
    _validate_move(tour, left, right)
    _validate_distance_matrix(distances)

    city_before_segment = int(tour[left - 1])
    first_segment_city = int(tour[left])

    last_segment_city = int(tour[right])
    city_after_segment = int(tour[right + 1])

    removed_length = (
        distances[city_before_segment, first_segment_city]
        + distances[last_segment_city, city_after_segment]
    )

    added_length = (
        distances[city_before_segment, last_segment_city]
        + distances[first_segment_city, city_after_segment]
    )

    return int(added_length - removed_length)


def _validate_move(
    tour: IntArray,
    left: int,
    right: int,
) -> None:
    """Validate one proposed 2-opt segment."""
    if tour.ndim != 1:
        raise ValueError("Tour must be one-dimensional.")

    if len(tour) < 4:
        raise ValueError(
            "A closed tour must contain at least three cities."
        )

    if tour[0] != tour[-1]:
        raise ValueError("Tour must return to its starting city.")

    if not 1 <= left < right <= len(tour) - 2:
        raise ValueError(
            "2-opt indices must select a valid interior segment."
        )


def _validate_distance_matrix(
    distances: IntArray,
) -> None:
    """Validate basic distance-matrix structure."""
    if distances.ndim != 2:
        raise ValueError("Distance matrix must be two-dimensional.")

    if distances.shape[0] != distances.shape[1]:
        raise ValueError("Distance matrix must be square.")

    if np.any(distances < 0):
        raise ValueError("Distances cannot be negative.")