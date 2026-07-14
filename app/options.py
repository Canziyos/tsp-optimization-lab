"""CLI options and typed solver configurations."""

import argparse

from algorithms.ant_colony import ACOConfig
from algorithms.genetic import GAConfig
from algorithms.hybrid import HybridConfig
from algorithms.two_opt import TwoOptConfig


def add_progress_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        metavar="N",
        help="Print iterative solver progress every N steps (default: 50).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress messages and print only the final result.",
    )


def add_ga_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--population", type=int, default=200)
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--crossover", type=float, default=0.80)
    parser.add_argument("--mutation", type=float, default=0.05)
    parser.add_argument("--tournament", type=int, default=4)
    parser.add_argument("--win-probability", type=float, default=0.75)
    parser.add_argument("--elites", type=int, default=10)
    parser.add_argument("--target", type=int)


def add_aco_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ants", type=int, default=52)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--evaporation", type=float, default=0.10)
    parser.add_argument("--deposit-weight", type=float, default=100.0)
    parser.add_argument("--ranked-ants", type=int, default=10)
    parser.add_argument("--elite-weight", type=float, default=0.0)


def add_hybrid_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hybrid-crossover", type=float, default=0.90)
    parser.add_argument("--hybrid-mutation", type=float, default=0.20)
    parser.add_argument("--hybrid-elites", type=int, default=2)
    parser.add_argument("--refine-interval", type=int, default=10)
    parser.add_argument("--refine-count", type=int, default=10)
    parser.add_argument("--refine-moves", type=int, default=20)


def _progress_interval(args: argparse.Namespace) -> int:
    return 0 if args.quiet else args.progress_every


def ga_config(args: argparse.Namespace, seed: int) -> GAConfig:
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
        progress_every=_progress_interval(args),
    )


def aco_config(args: argparse.Namespace, seed: int) -> ACOConfig:
    return ACOConfig(
        ant_count=args.ants,
        iterations=args.iterations,
        alpha=args.alpha,
        beta=args.beta,
        evaporation=args.evaporation,
        deposit_weight=args.deposit_weight,
        ranked_ants=args.ranked_ants,
        elite_weight=args.elite_weight,
        seed=seed,
        target_length=args.target,
        progress_every=_progress_interval(args),
    )


def hybrid_config(args: argparse.Namespace, seed: int) -> HybridConfig:
    genetic = GAConfig(
        population_size=args.population,
        generations=args.generations,
        crossover_probability=args.hybrid_crossover,
        mutation_probability=args.hybrid_mutation,
        tournament_size=args.tournament,
        tournament_win_probability=args.win_probability,
        elite_count=args.hybrid_elites,
        seed=seed,
        target_length=args.target,
        progress_every=_progress_interval(args),
    )

    two_opt = TwoOptConfig(
        max_passes=args.refine_moves,
        first_improvement=True,
    )

    return HybridConfig(
        genetic=genetic,
        two_opt=two_opt,
        refine_every=args.refine_interval,
        refine_count=args.refine_count,
        target_length=args.target,
    )
