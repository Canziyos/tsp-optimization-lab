"""Memetic genetic algorithm with periodic 2-opt refinement."""

from dataclasses import dataclass

import numpy as np

from .genetic import GAConfig, _solve_genetic
from .models import SolverResult
from .two_opt import best_two_opt_move, solve_two_opt


@dataclass(frozen=True, slots=True)
class HybridConfig:
    ga: GAConfig = GAConfig(
        crossover_probability=0.90,
        mutation_probability=0.20,
        elite_count=2,
    )
    refine_interval: int = 10
    refine_count: int = 10
    refine_moves: int | None = None

    def __post_init__(self) -> None:
        if self.refine_interval < 1 or self.refine_count < 1:
            raise ValueError("Refinement interval and count must be positive.")
        if self.refine_moves is not None and self.refine_moves < 1:
            raise ValueError("refine_moves must be positive when provided.")


def _refiner(config: HybridConfig):
    def refine(population, lengths, distances, step: int) -> None:
        if step % config.refine_interval:
            return
        selected = np.argsort(lengths)[: min(config.refine_count, len(population))]
        for index in selected:
            moves = 0
            while config.refine_moves is None or moves < config.refine_moves:
                delta, left, right = best_two_opt_move(population[index], distances)
                if delta >= 0:
                    break
                population[index, left : right + 1] = \
                    population[index, left : right + 1][::-1]
                lengths[index] += delta
                moves += 1
    return refine


def solve_hybrid(
    coordinates: np.ndarray, config: HybridConfig = HybridConfig()
) -> SolverResult:
    seed_tour = solve_two_opt(coordinates).best_tour
    return _solve_genetic(
        coordinates,
        config.ga,
        "hybrid-ga-two-opt",
        initial_tour=seed_tour,
        refiner=_refiner(config),
    )
