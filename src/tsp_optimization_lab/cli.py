"""Command-line interface for solving and benchmarking TSP methods."""

import argparse
import json
from pathlib import Path

from .ant_colony import solve_ant_colony
from .benchmark import run_benchmark
from .cli_options import (
    aco_config, add_aco_options, add_ga_options, add_hybrid_options,
    ga_config, hybrid_config,
)
from .genetic import solve_genetic
from .hybrid import solve_hybrid
from .nearest import solve_nearest_neighbor
from .reporting import write_benchmark_csv, write_history_csv, write_history_plot
from .tsplib import load_tsplib
from .two_opt import solve_two_opt


def _parser() -> argparse.ArgumentParser:
    data = Path(__file__).resolve().parent / "data" / "berlin52.tsp"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=data)
    commands = parser.add_subparsers(dest="command", required=True)

    solve = commands.add_parser("solve", help="Run one solver.")
    solve.add_argument(
        "algorithm", choices=("nearest", "two-opt", "ga", "aco", "hybrid")
    )
    solve.add_argument("--seed", type=int, default=2)
    solve.add_argument("--csv", type=Path)
    solve.add_argument("--plot", type=Path)
    add_ga_options(solve)
    add_aco_options(solve)
    add_hybrid_options(solve)

    benchmark = commands.add_parser("benchmark", help="Compare available solvers.")
    benchmark.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    benchmark.add_argument("--csv", type=Path)
    add_ga_options(benchmark)
    add_aco_options(benchmark)
    add_hybrid_options(benchmark)
    return parser


def _solve(args, coordinates):
    if args.algorithm == "nearest":
        result = solve_nearest_neighbor(coordinates)
    elif args.algorithm == "two-opt":
        result = solve_two_opt(coordinates)
    elif args.algorithm == "ga":
        result = solve_genetic(coordinates, ga_config(args, args.seed))
    elif args.algorithm == "aco":
        result = solve_ant_colony(coordinates, aco_config(args, args.seed))
    else:
        result = solve_hybrid(coordinates, hybrid_config(args, args.seed))
    if args.csv:
        write_history_csv(result, args.csv)
    if args.plot:
        write_history_plot(result, args.plot)
    return {
        "algorithm": result.algorithm,
        "best_length": result.best_length,
        "best_step": result.best_step,
        "seed": result.seed,
        "tour": result.best_tour.tolist(),
    }


def _benchmark(args, coordinates):
    runs = run_benchmark(
        coordinates, tuple(args.seeds), ga_config(args, args.seeds[0]),
        aco_config(args, args.seeds[0]), hybrid_config(args, args.seeds[0]),
    )
    if args.csv:
        write_benchmark_csv(runs, args.csv)
    return [{
        "algorithm": run.algorithm,
        "seed": run.seed,
        "best_length": run.result.best_length,
        "best_step": run.result.best_step,
        "seconds": round(run.elapsed_seconds, 6),
    } for run in runs]


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    coordinates, metadata = load_tsplib(args.data)
    output = _solve(args, coordinates) if args.command == "solve" \
        else _benchmark(args, coordinates)
    print(json.dumps({"instance": metadata.get("NAME"), "results": output}, indent=2))
