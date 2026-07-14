"""2-opt local-search solver."""

import numpy as np
from numpy.typing import NDArray

from core.distances import distance_matrix
from core.models import HistoryPoint, SolverResult
from core.tours import random_tour, tour_length, validate_tour

from .config import TwoOptConfig
from .moves import apply_two_opt_move, two_opt_delta


IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


def improve_two_opt(
    tour: IntArray,
    distances: IntArray,
    config: TwoOptConfig = TwoOptConfig(),
) -> tuple[IntArray, int, tuple[HistoryPoint, ...]]:
    """
    Improve an existing tour using repeated 2-opt moves.

    Returns the improved tour, its length, and optimization history.
    """
    city_count = distances.shape[0]

    validate_tour(
        tour=tour,
        city_count=city_count,
    )

    current_tour = tour.copy()
    current_length = tour_length(
        tour=current_tour,
        distances=distances,
    )

    history: list[HistoryPoint] = [
        HistoryPoint(
            step=0,
            best_length=current_length,
            mean_length=float(current_length),
        )
    ]

    for pass_number in range(1, config.max_passes + 1):
        move = _find_improving_move(
            tour=current_tour,
            distances=distances,
            first_improvement=config.first_improvement,
        )

        if move is None:
            break

        left, right, delta = move

        current_tour = apply_two_opt_move(
            tour=current_tour,
            left=left,
            right=right,
        )

        current_length += delta

        history.append(
            HistoryPoint(
                step=pass_number,
                best_length=current_length,
                mean_length=float(current_length),
            )
        )

        if (
            config.target_length is not None
            and current_length <= config.target_length
        ):
            break

    return current_tour, current_length, tuple(history)


def solve_two_opt(
    coordinates: FloatArray,
    config: TwoOptConfig = TwoOptConfig(),
    initial_tour: IntArray | None = None,
    seed: int = 2,
) -> SolverResult:
    """Solve a TSP instance using 2-opt local search."""
    distances = distance_matrix(coordinates)
    city_count = len(coordinates)

    if city_count < 3:
        raise ValueError("At least three cities are required.")

    if initial_tour is None:
        rng = np.random.default_rng(seed)

        initial_tour = random_tour(
            city_count=city_count,
            rng=rng,
        )
    else:
        validate_tour(
            tour=initial_tour,
            city_count=city_count,
        )

    best_tour, best_length, history = improve_two_opt(
        tour=initial_tour,
        distances=distances,
        config=config,
    )

    return SolverResult(
        algorithm="two-opt",
        best_tour=best_tour,
        best_length=best_length,
        best_step=history[-1].step,
        history=history,
        seed=seed,
    )


def _find_improving_move(
    tour: IntArray,
    distances: IntArray,
    first_improvement: bool,
) -> tuple[int, int, int] | None:
    """
    Find either the first or best improving 2-opt move.

    Returns:
        left index, right index, and negative length delta.
    """
    best_move: tuple[int, int, int] | None = None
    best_delta = 0

    last_interior_index = len(tour) - 2

    for left in range(1, last_interior_index):
        for right in range(left + 1, last_interior_index + 1):
            delta = two_opt_delta(
                tour=tour,
                distances=distances,
                left=left,
                right=right,
            )

            if delta >= 0:
                continue

            if first_improvement:
                return left, right, delta

            if delta < best_delta:
                best_delta = delta
                best_move = left, right, delta

    return best_move