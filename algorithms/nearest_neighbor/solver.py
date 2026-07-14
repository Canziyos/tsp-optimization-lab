"""Nearest-neighbor TSP solver."""

import numpy as np
from numpy.typing import NDArray

from core.distances import distance_matrix
from core.models import HistoryPoint, SolverResult
from core.tours import tour_length

from .config import NearestNeighborConfig


IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


def build_nearest_neighbor_tour(
    distances: IntArray,
    start_city: int = 0,
) -> IntArray:
    """
    Build one closed tour by repeatedly choosing the nearest unvisited city.
    """
    _validate_distance_matrix(distances)

    city_count = distances.shape[0]

    if not 0 <= start_city < city_count:
        raise ValueError("Start city is outside the valid city range.")

    tour = np.empty(city_count + 1, dtype=np.int64)
    tour[0] = start_city
    tour[-1] = start_city

    unvisited = np.ones(city_count, dtype=np.bool_)
    unvisited[start_city] = False

    current_city = start_city

    for position in range(1, city_count):
        candidates = np.flatnonzero(unvisited)

        candidate_distances = distances[current_city, candidates]

        nearest_position = int(np.argmin(candidate_distances))
        next_city = int(candidates[nearest_position])

        tour[position] = next_city
        unvisited[next_city] = False
        current_city = next_city

    return tour


def solve_nearest_neighbor(
    coordinates: FloatArray,
    config: NearestNeighborConfig = NearestNeighborConfig(),
) -> SolverResult:
    """Solve a symmetric TSP instance using nearest-neighbor search."""
    distances = distance_matrix(coordinates)
    city_count = len(coordinates)

    if city_count < 3:
        raise ValueError("At least three cities are required.")

    if config.start_city >= city_count:
        raise ValueError("Start city is outside the valid city range.")

    if config.multi_start:
        best_tour, best_length, best_start = _solve_multi_start(
            distances=distances,
        )
    else:
        best_tour = build_nearest_neighbor_tour(
            distances=distances,
            start_city=config.start_city,
        )

        best_length = tour_length(
            tour=best_tour,
            distances=distances,
        )

        best_start = config.start_city

    history = (
        HistoryPoint(
            step=best_start,
            best_length=best_length,
            mean_length=float(best_length),
        ),
    )

    return SolverResult(
        algorithm="nearest-neighbor",
        best_tour=best_tour,
        best_length=best_length,
        best_step=best_start,
        history=history,
        seed=None,
    )



def normalize_tour_start(
    tour: IntArray,
    start_city: int = 0,
) -> IntArray:
    """Rotate a closed tour without changing its cycle."""
    if tour.ndim != 1 or len(tour) < 2 or tour[0] != tour[-1]:
        raise ValueError("Tour must be a closed one-dimensional tour.")
    interior = tour[:-1]
    matches = np.flatnonzero(interior == start_city)
    if len(matches) != 1:
        raise ValueError("Requested start city must occur exactly once.")
    rotated = np.roll(interior, -int(matches[0]))
    return np.concatenate((rotated, rotated[:1])).astype(np.int64, copy=False)


def _solve_multi_start(
    distances: IntArray,
) -> tuple[IntArray, int, int]:
    """
    Run nearest neighbor from every possible start city.

    Return the best tour, its length, and the winning start city.
    """
    city_count = distances.shape[0]

    best_tour: IntArray | None = None
    best_length = np.iinfo(np.int64).max
    best_start = 0

    for start_city in range(city_count):
        tour = build_nearest_neighbor_tour(
            distances=distances,
            start_city=start_city,
        )

        tour = normalize_tour_start(tour, start_city=0)

        length = tour_length(
            tour=tour,
            distances=distances,
        )

        if length < best_length:
            best_tour = tour.copy()
            best_length = length
            best_start = start_city

    if best_tour is None:
        raise RuntimeError("Nearest-neighbor search produced no tour.")

    return best_tour, int(best_length), best_start


def _validate_distance_matrix(
    distances: IntArray,
) -> None:
    """Validate the distance matrix used by nearest-neighbor search."""
    if distances.ndim != 2:
        raise ValueError("Distance matrix must be two-dimensional.")

    row_count, column_count = distances.shape

    if row_count != column_count:
        raise ValueError("Distance matrix must be square.")

    if row_count < 2:
        raise ValueError("At least two cities are required.")

    if np.any(distances < 0):
        raise ValueError("Distances cannot be negative.")

    if not np.all(np.isfinite(distances)):
        raise ValueError("Distance matrix contains non-finite values.")

    if not np.array_equal(distances, distances.T):
        raise ValueError("Distance matrix must be symmetric.")

    if np.any(np.diag(distances) != 0):
        raise ValueError("Distance matrix diagonal must be zero.")