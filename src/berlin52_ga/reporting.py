"""Explicit result writers; the solver itself never creates files."""

import csv
from pathlib import Path

from .solver import GAResult


def write_history_csv(result: GAResult, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("generation", "best_length", "average_length"))
        writer.writerows(result.history)


def write_history_plot(result: GAResult, path: str | Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install berlin52-ga[plot] to create plots.") from error

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    generations, best, average = zip(*result.history)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(generations, best, label="Best")
    axis.plot(generations, average, label="Population average", alpha=0.75)
    axis.set(xlabel="Generation", ylabel="TSPLIB EUC_2D length")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)

