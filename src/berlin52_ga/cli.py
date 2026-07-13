"""Command-line entry point for reproducible Berlin52 experiments."""

import argparse
import json
from pathlib import Path

from .reporting import write_history_csv, write_history_plot
from .solver import GAConfig, run_ga
from .tsplib import load_tsplib


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = Path(__file__).resolve().parent / "data" / "berlin52.tsp"
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--population", type=int, default=200)
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--crossover", type=float, default=0.80)
    parser.add_argument("--mutation", type=float, default=0.05)
    parser.add_argument("--tournament", type=int, default=4)
    parser.add_argument("--win-probability", type=float, default=0.75)
    parser.add_argument("--elites", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--target", type=int)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--plot", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    coordinates, metadata = load_tsplib(args.data)
    config = GAConfig(
        population_size=args.population,
        generations=args.generations,
        crossover_probability=args.crossover,
        mutation_probability=args.mutation,
        tournament_size=args.tournament,
        tournament_win_probability=args.win_probability,
        elite_count=args.elites,
        seed=args.seed,
        target_length=args.target,
    )
    result = run_ga(coordinates, config)
    if args.csv:
        write_history_csv(result, args.csv)
    if args.plot:
        write_history_plot(result, args.plot)
    print(json.dumps({
        "instance": metadata.get("NAME", args.data.stem),
        "best_length": result.best_length,
        "best_generation": result.best_generation,
        "generations_run": len(result.history) - 1,
        "seed": config.seed,
        "tour": result.best_tour.tolist(),
    }, indent=2))
