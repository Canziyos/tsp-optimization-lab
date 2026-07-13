"""Command-line interface for solving and benchmarking TSP methods."""

import argparse
import json
from pathlib import Path

from .benchmark import run_benchmark
from .genetic import GAConfig, solve_genetic
from .nearest import solve_nearest_neighbor
from .reporting import write_benchmark_csv, write_history_csv, write_history_plot
from .tsplib import load_tsplib
from .two_opt import solve_two_opt


def _add_ga_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--population", type=int, default=200)
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--crossover", type=float, default=0.80)
    parser.add_argument("--mutation", type=float, default=0.05)
    parser.add_argument("--tournament", type=int, default=4)
    parser.add_argument("--win-probability", type=float, default=0.75)
    parser.add_argument("--elites", type=int, default=10)
    parser.add_argument("--target", type=int)


def _parser() -> argparse.ArgumentParser:
    data = Path(__file__).resolve().parent / "data" / "berlin52.tsp"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=data)
    commands = parser.add_subparsers(dest="command", required=True)

    solve = commands.add_parser("solve", help="Run one solver.")
    solve.add_argument("algorithm", choices=("nearest", "two-opt", "ga"))
    solve.add_argument("--seed", type=int, default=2)
    solve.add_argument("--csv", type=Path)
    solve.add_argument("--plot", type=Path)
    _add_ga_options(solve)

    benchmark = commands.add_parser("benchmark", help="Compare available solvers.")
    benchmark.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    benchmark.add_argument("--csv", type=Path)
    _add_ga_options(benchmark)
    return parser


def _ga_config(args, seed: int) -> GAConfig:
    return GAConfig(
        population_size=args.population,
        generations=args.generations,
        crossover_probability=args.crossover,
        mutation_probability=args.mutation,
        tournament_size=args.tournament,
        tournament_win_probability=args.win_probability,
        elite_count=args.elites,
        seed=seed,
        target_length=args.target,
    )


def _solve(args, coordinates):
    if args.algorithm == "nearest":
        result = solve_nearest_neighbor(coordinates)
    elif args.algorithm == "two-opt":
        result = solve_two_opt(coordinates)
    else:
        result = solve_genetic(coordinates, _ga_config(args, args.seed))
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
        coordinates, tuple(args.seeds), _ga_config(args, args.seeds[0])
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

