"""High-level ant-colony optimization solver."""

import numpy as np
from numpy.typing import NDArray

from core.distances import distance_matrix
from core.models import HistoryPoint, SolverResult

from .colony import ColonyResult, run_colony_iteration
from .config import ACOConfig
from .pheromone import PheromoneMap


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def build_heuristic(distances: IntArray) -> FloatArray:
    """
    Build inverse-distance heuristic information.

    Closer cities receive larger heuristic values.
    """
    if distances.ndim != 2:
        raise ValueError("Distance matrix must be two-dimensional.")

    row_count, column_count = distances.shape

    if row_count != column_count:
        raise ValueError("Distance matrix must be square.")

    if np.any(distances < 0):
        raise ValueError("Distances cannot be negative.")

    heuristic = np.zeros_like(
        distances,
        dtype=np.float64,
    )

    np.divide(
        1.0,
        distances,
        out=heuristic,
        where=distances > 0,
    )

    np.fill_diagonal(heuristic, 0.0)

    return heuristic


def update_global_best(
    colony: ColonyResult,
    best_tour: IntArray | None,
    best_length: int,
    best_step: int,
    current_step: int,
) -> tuple[IntArray, int, int]:
    """Update the best solution found across all iterations."""
    if best_tour is None or colony.best_length < best_length:
        return (
            colony.best_tour.copy(),
            colony.best_length,
            current_step,
        )

    return best_tour, best_length, best_step


def update_pheromones(
    pheromone: PheromoneMap,
    colony: ColonyResult,
    best_tour: IntArray,
    best_length: int,
    config: ACOConfig,
) -> None:
    """Evaporate old pheromone and reinforce successful tours."""
    pheromone.evaporate(config.evaporation)

    ranked_count = min(
        config.ranked_ants,
        config.ant_count,
    )

    pheromone.deposit_ranked(
        tours=colony.tours,
        lengths=colony.lengths,
        ranked_ants=ranked_count,
        deposit_weight=config.deposit_weight,
    )

    if config.elite_weight > 0.0:
        pheromone.deposit_by_length(
            tour=best_tour,
            tour_length=best_length,
            deposit_weight=(
                config.elite_weight
                * config.deposit_weight
            ),
        )


def solve_ant_colony(
    coordinates: FloatArray,
    config: ACOConfig = ACOConfig(),
) -> SolverResult:
    """Solve a symmetric TSP instance using ant-colony optimization."""
    distances = distance_matrix(coordinates)
    city_count = len(distances)

    if city_count < 3:
        raise ValueError("At least three cities are required.")

    heuristic = build_heuristic(distances)

    pheromone = PheromoneMap.uniform(
        city_count=city_count,
        initial_value=1.0,
    )

    rng = np.random.default_rng(config.seed)

    best_tour: IntArray | None = None
    best_length = np.iinfo(np.int64).max
    best_step = 0

    history: list[HistoryPoint] = []

    for step in range(1, config.iterations + 1):
        colony = run_colony_iteration(
            pheromone=pheromone,
            heuristic=heuristic,
            distances=distances,
            config=config,
            rng=rng,
        )

        best_tour, best_length, best_step = update_global_best(
            colony=colony,
            best_tour=best_tour,
            best_length=int(best_length),
            best_step=best_step,
            current_step=step,
        )

        history.append(
            HistoryPoint(
                step=step,
                best_length=best_length,
                mean_length=colony.mean_length,
            )
        )

        _report_progress(
            step=step,
            total_steps=config.iterations,
            best_length=best_length,
            mean_length=colony.mean_length,
            progress_every=config.progress_every,
            force=step == 1 or step == config.iterations,
        )

        update_pheromones(
            pheromone=pheromone,
            colony=colony,
            best_tour=best_tour,
            best_length=best_length,
            config=config,
        )

        if (
            config.target_length is not None
            and best_length <= config.target_length
        ):
            break

    if best_tour is None:
        raise RuntimeError("Ant colony produced no solution.")

    return SolverResult(
        algorithm="ant-colony",
        best_tour=best_tour,
        best_length=best_length,
        best_step=best_step,
        history=tuple(history),
        seed=config.seed,
    )

def _report_progress(
    step: int,
    total_steps: int,
    best_length: int,
    mean_length: float,
    progress_every: int,
    force: bool = False,
) -> None:
    """Print a compact ACO progress line when reporting is enabled."""
    if progress_every <= 0:
        return

    if not force and step % progress_every != 0:
        return

    print(
        f"[ACO] iteration {step}/{total_steps} | "
        f"best={best_length} | mean={mean_length:.1f}",
        flush=True,
    )
