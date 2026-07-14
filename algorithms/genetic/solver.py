"""High-level genetic TSP solver."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from core.distances import distance_matrix
from core.models import HistoryPoint, SolverResult
from core.tours import population_lengths, validate_tour

from .config import GAConfig
from .population import (
    PopulationState,
    create_population,
    evaluate_population,
    replace_worst,
    select_survivors,
)
from .reproduction import create_offspring


IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]

PopulationRefiner = Callable[
    [IntArray, IntArray, IntArray, int],
    None,
]


def solve_genetic(
    coordinates: FloatArray,
    config: GAConfig = GAConfig(),
) -> SolverResult:
    """Solve a symmetric TSP using a genetic algorithm."""
    return solve_genetic_engine(
        coordinates=coordinates,
        config=config,
        algorithm_name="genetic-algorithm",
    )


def solve_genetic_engine(
    coordinates: FloatArray,
    config: GAConfig,
    algorithm_name: str,
    initial_tour: IntArray | None = None,
    refiner: PopulationRefiner | None = None,
) -> SolverResult:
    """
    Run the genetic engine.

    The optional initial tour and refiner are extension points used by
    hybrid algorithms.
    """
    distances = distance_matrix(coordinates)
    city_count = len(coordinates)

    if city_count < 3:
        raise ValueError("At least three cities are required.")

    rng = np.random.default_rng(config.seed)

    population = create_population(
        population_size=config.population_size,
        city_count=city_count,
        distances=distances,
        rng=rng,
    )

    if initial_tour is not None:
        population = _insert_initial_tour(
            population=population,
            initial_tour=initial_tour,
            distances=distances,
            city_count=city_count,
        )

    best_tour = population.best_tour.copy()
    best_length = population.best_length
    best_step = 0

    history: list[HistoryPoint] = [
        HistoryPoint(
            step=0,
            best_length=best_length,
            mean_length=population.mean_length,
        )
    ]

    _report_progress(
        algorithm_name=algorithm_name,
        step=0,
        total_steps=config.generations,
        best_length=best_length,
        mean_length=population.mean_length,
        progress_every=config.progress_every,
        force=True,
    )

    for generation in range(1, config.generations + 1):
        population = _run_generation(
            population=population,
            distances=distances,
            config=config,
            rng=rng,
            generation=generation,
            refiner=refiner,
        )

        if population.best_length < best_length:
            best_tour = population.best_tour.copy()
            best_length = population.best_length
            best_step = generation

        history.append(
            HistoryPoint(
                step=generation,
                best_length=population.best_length,
                mean_length=population.mean_length,
            )
        )

        _report_progress(
            algorithm_name=algorithm_name,
            step=generation,
            total_steps=config.generations,
            best_length=best_length,
            mean_length=population.mean_length,
            progress_every=config.progress_every,
            force=generation == config.generations,
        )

        if (
            config.target_length is not None
            and best_length <= config.target_length
        ):
            break

    return SolverResult(
        algorithm=algorithm_name,
        best_tour=best_tour,
        best_length=best_length,
        best_step=best_step,
        history=tuple(history),
        seed=config.seed,
    )


def _run_generation(
    population: PopulationState,
    distances: IntArray,
    config: GAConfig,
    rng: np.random.Generator,
    generation: int,
    refiner: PopulationRefiner | None,
) -> PopulationState:
    """Create, optionally refine, and select one new generation."""
    child_tours = create_offspring(
        population=population,
        config=config,
        rng=rng,
    )

    child_lengths = population_lengths(
        population=child_tours,
        distances=distances,
    ).astype(np.int64, copy=False)

    if refiner is not None:
        refiner(
            child_tours,
            child_lengths,
            distances,
            generation,
        )

    children = PopulationState(
        tours=child_tours,
        lengths=child_lengths,
    )

    return select_survivors(
        current=population,
        children=children,
        population_size=config.population_size,
        elite_count=config.elite_count,
    )


def _insert_initial_tour(
    population: PopulationState,
    initial_tour: IntArray,
    distances: IntArray,
    city_count: int,
) -> PopulationState:
    """Insert a supplied tour by replacing the worst random individual."""
    validate_tour(
        tour=initial_tour,
        city_count=city_count,
    )

    initial_length = int(
        population_lengths(
            population=initial_tour[np.newaxis, :],
            distances=distances,
        )[0]
    )

    return replace_worst(
        population=population,
        tour=initial_tour,
        tour_length=initial_length,
    )

def _report_progress(
    algorithm_name: str,
    step: int,
    total_steps: int,
    best_length: int,
    mean_length: float,
    progress_every: int,
    force: bool = False,
) -> None:
    """Print a compact progress line when progress reporting is enabled."""
    if progress_every <= 0:
        return

    if not force and step % progress_every != 0:
        return

    label = "Hybrid" if algorithm_name.startswith("hybrid") else "Genetic"
    print(
        f"[{label}] generation {step}/{total_steps} | "
        f"best={best_length} | mean={mean_length:.1f}",
        flush=True,
    )
