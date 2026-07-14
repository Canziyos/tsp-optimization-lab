"""Comparable quality and runtime measurements for available solvers."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from algorithms.ant_colony import ACOConfig, solve_ant_colony
from algorithms.genetic import GAConfig, solve_genetic
from algorithms.hybrid import HybridConfig, solve_hybrid
from algorithms.nearest_neighbor import solve_nearest_neighbor
from algorithms.two_opt import solve_two_opt
from core.models import SolverResult


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    algorithm: str
    seed: int | None
    elapsed_seconds: float
    result: SolverResult


def _timed(
    solver: Callable[[], SolverResult],
    algorithm: str,
    seed: int | None,
    show_progress: bool,
) -> BenchmarkRun:
    label = algorithm if seed is None else f"{algorithm} (seed={seed})"

    if show_progress:
        print(f"\n[Benchmark] starting {label}...", flush=True)

    started = perf_counter()
    result = solver()
    elapsed = perf_counter() - started

    if show_progress:
        print(
            f"[Benchmark] finished {label} | "
            f"best={result.best_length} | "
            f"step={result.best_step} | "
            f"time={elapsed:.3f}s",
            flush=True,
        )

    return BenchmarkRun(
        algorithm=algorithm,
        seed=seed,
        elapsed_seconds=elapsed,
        result=result,
    )


def run_benchmark(
    coordinates: np.ndarray,
    seeds: tuple[int, ...] = (0, 1, 2),
    ga_config: GAConfig = GAConfig(),
    aco_config: ACOConfig = ACOConfig(),
    hybrid_config: HybridConfig = HybridConfig(),
    show_progress: bool = False,
) -> tuple[BenchmarkRun, ...]:
    if not seeds:
        raise ValueError("At least one stochastic-solver seed is required.")

    nearest = _timed(
        solver=lambda: solve_nearest_neighbor(coordinates),
        algorithm="nearest-neighbor",
        seed=None,
        show_progress=show_progress,
    )

    two_opt = _timed(
        solver=lambda: solve_two_opt(
            coordinates=coordinates,
            initial_tour=nearest.result.best_tour,
        ),
        algorithm="two-opt",
        seed=None,
        show_progress=show_progress,
    )

    runs = [nearest, two_opt]

    for seed in seeds:
        config = replace(ga_config, seed=seed)
        runs.append(
            _timed(
                solver=lambda config=config: solve_genetic(coordinates, config),
                algorithm="genetic-algorithm",
                seed=seed,
                show_progress=show_progress,
            )
        )

    for seed in seeds:
        config = replace(aco_config, seed=seed)
        runs.append(
            _timed(
                solver=lambda config=config: solve_ant_colony(coordinates, config),
                algorithm="ant-colony",
                seed=seed,
                show_progress=show_progress,
            )
        )

    for seed in seeds:
        config = replace(
            hybrid_config,
            genetic=replace(hybrid_config.genetic, seed=seed),
        )
        runs.append(
            _timed(
                solver=lambda config=config: solve_hybrid(coordinates, config),
                algorithm="hybrid-genetic-two-opt",
                seed=seed,
                show_progress=show_progress,
            )
        )

    return tuple(runs)
