"""Hybrid genetic and local-search TSP solver."""

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from algorithms.genetic.solver import solve_genetic_engine
from algorithms.nearest_neighbor import build_nearest_neighbor_tour
from algorithms.two_opt import improve_two_opt

from core.distances import distance_matrix
from core.models import SolverResult

from .config import HybridConfig
from .refinement import refine_best_tours


IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


def solve_hybrid(
    coordinates: FloatArray,
    config: HybridConfig = HybridConfig(),
) -> SolverResult:
    """
    Solve a TSP instance using nearest-neighbor initialization,
    genetic exploration, and 2-opt local refinement.
    """
    distances = distance_matrix(coordinates)
    city_count = len(coordinates)

    if city_count < 3:
        raise ValueError("At least three cities are required.")

    initial_tour = _build_initial_tour(
        distances=distances,
        enabled=config.use_nearest_neighbor_seed,
    )

    genetic_config = replace(
        config.genetic,
        target_length=config.target_length,
    )

    def population_refiner(
        tours: IntArray,
        lengths: IntArray,
        distance_matrix_: IntArray,
        generation: int,
    ) -> None:
        if generation % config.refine_every != 0:
            return

        refine_best_tours(
            tours=tours,
            lengths=lengths,
            distances=distance_matrix_,
            refine_count=config.refine_count,
            config=config.two_opt,
        )

    genetic_result = solve_genetic_engine(
        coordinates=coordinates,
        config=genetic_config,
        algorithm_name="hybrid-genetic-two-opt",
        initial_tour=initial_tour,
        refiner=population_refiner,
    )

    final_tour, final_length, final_history = improve_two_opt(
        tour=genetic_result.best_tour,
        distances=distances,
        config=config.two_opt,
    )

    if final_length >= genetic_result.best_length:
        return genetic_result

    generation_offset = genetic_result.history[-1].step

    appended_history = tuple(
        replace(
            point,
            step=generation_offset + point.step,
        )
        for point in final_history[1:]
    )

    combined_history = (
        genetic_result.history
        + appended_history
    )

    return SolverResult(
        algorithm="hybrid-genetic-two-opt",
        best_tour=final_tour,
        best_length=final_length,
        best_step=combined_history[-1].step,
        history=combined_history,
        seed=genetic_result.seed,
    )


def _build_initial_tour(
    distances: IntArray,
    enabled: bool,
) -> IntArray | None:
    """Optionally create a nearest-neighbor seed tour."""
    if not enabled:
        return None

    return build_nearest_neighbor_tour(
        distances=distances,
        start_city=0,
    )