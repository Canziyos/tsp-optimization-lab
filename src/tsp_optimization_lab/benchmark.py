"""Comparable quality and runtime measurements for available solvers."""

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from .genetic import GAConfig, solve_genetic
from .models import SolverResult
from .nearest import solve_nearest_neighbor
from .two_opt import solve_two_opt


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    algorithm: str
    seed: int | None
    elapsed_seconds: float
    result: SolverResult


def _timed(solver, algorithm: str, seed: int | None) -> BenchmarkRun:
    started = perf_counter()
    result = solver()
    return BenchmarkRun(algorithm, seed, perf_counter() - started, result)


def run_benchmark(
    coordinates: np.ndarray,
    seeds: tuple[int, ...] = (0, 1, 2),
    ga_config: GAConfig = GAConfig(),
) -> tuple[BenchmarkRun, ...]:
    if not seeds:
        raise ValueError("At least one GA seed is required.")
    nearest = _timed(
        lambda: solve_nearest_neighbor(coordinates), "nearest-neighbor", None
    )
    two_opt = _timed(
        lambda: solve_two_opt(coordinates, nearest.result.best_tour), "two-opt", None
    )
    runs = [nearest, two_opt]
    for seed in seeds:
        config = replace(ga_config, seed=seed)
        runs.append(_timed(
            lambda config=config: solve_genetic(coordinates, config),
            "genetic-algorithm",
            seed,
        ))
    return tuple(runs)

