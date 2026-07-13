"""Best-improvement 2-opt local search."""

import numpy as np

from .models import HistoryPoint, SolverResult
from .nearest import solve_nearest_neighbor
from .tours import tour_length, validate_tour
from .tsplib import distance_matrix


def best_two_opt_move(
    tour: np.ndarray, distances: np.ndarray
) -> tuple[int, int, int]:
    best_delta, best_left, best_right = 0, 0, 0
    for left in range(1, len(tour) - 2):
        for right in range(left + 1, len(tour) - 1):
            old = distances[tour[left - 1], tour[left]] \
                + distances[tour[right], tour[right + 1]]
            new = distances[tour[left - 1], tour[right]] \
                + distances[tour[left], tour[right + 1]]
            delta = int(new - old)
            if delta < best_delta:
                best_delta, best_left, best_right = delta, left, right
    return best_delta, best_left, best_right


def solve_two_opt(
    coordinates: np.ndarray,
    initial_tour: np.ndarray | None = None,
    max_moves: int | None = None,
) -> SolverResult:
    distances = distance_matrix(coordinates)
    if max_moves is not None and max_moves < 1:
        raise ValueError("max_moves must be positive when provided.")
    if initial_tour is None:
        initial_tour = solve_nearest_neighbor(coordinates).best_tour
    validate_tour(initial_tour, len(coordinates))
    tour = np.asarray(initial_tour, dtype=np.int64).copy()
    length = tour_length(tour, distances)
    history = [HistoryPoint(0, length)]

    moves = 0
    while max_moves is None or moves < max_moves:
        delta, left, right = best_two_opt_move(tour, distances)
        if delta >= 0:
            break
        tour[left : right + 1] = tour[left : right + 1][::-1]
        length += delta
        moves += 1
        history.append(HistoryPoint(moves, length))
    return SolverResult("two-opt", tour, length, moves, tuple(history))
