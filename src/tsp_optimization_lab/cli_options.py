"""CLI options and typed solver configurations."""

import argparse

from .ant_colony import ACOConfig
from .genetic import GAConfig
from .hybrid import HybridConfig


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
    parser.add_argument("--refine-moves", type=int)


def ga_config(args, seed: int) -> GAConfig:
    return GAConfig(
        args.population, args.generations, args.crossover, args.mutation,
        args.tournament, args.win_probability, args.elites, seed, args.target,
    )


def aco_config(args, seed: int) -> ACOConfig:
    return ACOConfig(
        args.ants, args.iterations, args.alpha, args.beta, args.evaporation,
        args.deposit_weight, args.ranked_ants, args.elite_weight, seed, args.target,
    )


def hybrid_config(args, seed: int) -> HybridConfig:
    ga = GAConfig(
        args.population, args.generations, args.hybrid_crossover,
        args.hybrid_mutation, args.tournament, args.win_probability,
        args.hybrid_elites, seed, args.target,
    )
    return HybridConfig(ga, args.refine_interval, args.refine_count, args.refine_moves)
