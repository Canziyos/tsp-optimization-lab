"""Explicit CSV and plot writers; solvers never create files."""

import csv
from pathlib import Path

from .benchmark import BenchmarkRun
from core.models import SolverResult


def write_history_csv(result: SolverResult, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("step", "best_length", "mean_length"))
        writer.writerows(
            (point.step, point.best_length, point.mean_length)
            for point in result.history
        )


def write_benchmark_csv(runs: tuple[BenchmarkRun, ...], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("algorithm", "seed", "best_length", "best_step", "seconds"))
        writer.writerows(
            (run.algorithm, run.seed, run.result.best_length,
             run.result.best_step, f"{run.elapsed_seconds:.6f}")
            for run in runs
        )


def write_history_plot(result: SolverResult, path: str | Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install tsp-optimization-lab[plot] to plot.") from error
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    steps = [point.step for point in result.history]
    best = [point.best_length for point in result.history]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(steps, best, label="Best")
    means = [point.mean_length for point in result.history]
    if all(value is not None for value in means):
        axis.plot(steps, means, label="Population average", alpha=0.75)
    axis.set(xlabel="Step", ylabel="TSPLIB EUC_2D length")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)

