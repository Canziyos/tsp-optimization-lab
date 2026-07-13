"""Genetic-algorithm configuration and solver."""

from dataclasses import dataclass

import numpy as np

from .models import HistoryPoint, SolverResult
from .operators import mutate_inversion, order_crossover, tournament_select
from .tours import initial_population, population_lengths, validate_tour
from .tsplib import distance_matrix


@dataclass(frozen=True, slots=True)
class GAConfig:
    population_size: int = 200
    generations: int = 1000
    crossover_probability: float = 0.80
    mutation_probability: float = 0.05
    tournament_size: int = 4
    tournament_win_probability: float = 0.75
    elite_count: int = 10
    seed: int = 2
    target_length: int | None = None

    def __post_init__(self) -> None:
        if self.population_size < 2 or self.generations < 1:
            raise ValueError("Population must be >= 2 and generations >= 1.")
        if not 0 <= self.elite_count < self.population_size:
            raise ValueError("elite_count must be within the population.")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("Invalid tournament_size.")
        probabilities = (
            self.crossover_probability,
            self.mutation_probability,
            self.tournament_win_probability,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("Probabilities must be in [0, 1].")


def _offspring(population, lengths, config, rng) -> np.ndarray:
    children: list[np.ndarray] = []
    while len(children) < config.population_size:
        parents = [
            tournament_select(
                population, lengths, config.tournament_size,
                config.tournament_win_probability, rng,
            )
            for _ in range(2)
        ]
        if rng.random() < config.crossover_probability:
            children += [
                order_crossover(parents[0], parents[1], rng),
                order_crossover(parents[1], parents[0], rng),
            ]
        else:
            children += parents
    return np.stack([
        mutate_inversion(child, config.mutation_probability, rng)
        for child in children[: config.population_size]
    ])


def _solve_genetic(
    coordinates: np.ndarray,
    config: GAConfig,
    algorithm: str,
    initial_tour: np.ndarray | None = None,
    refiner=None,
) -> SolverResult:
    distances = distance_matrix(coordinates)
    rng = np.random.default_rng(config.seed)
    population = initial_population(config.population_size, len(coordinates), rng)
    lengths = population_lengths(population, distances)
    if initial_tour is not None:
        validate_tour(initial_tour, len(coordinates))
        index = int(np.argmax(lengths))
        population[index] = initial_tour
        lengths[index] = population_lengths(population[index : index + 1], distances)[0]
    best_index = int(np.argmin(lengths))
    best_tour = population[best_index].copy()
    best_length, best_step = int(lengths[best_index]), 0
    history = [HistoryPoint(0, best_length, float(lengths.mean()))]

    for step in range(1, config.generations + 1):
        children = _offspring(population, lengths, config, rng)
        child_lengths = population_lengths(children, distances)
        if refiner is not None:
            refiner(children, child_lengths, distances, step)
        elite = np.argsort(lengths)[: config.elite_count]
        selected = np.argsort(child_lengths)[: config.population_size - config.elite_count]
        population = np.concatenate((population[elite], children[selected]))
        lengths = np.concatenate((lengths[elite], child_lengths[selected]))
        current = int(np.argmin(lengths))
        if lengths[current] < best_length:
            best_tour = population[current].copy()
            best_length, best_step = int(lengths[current]), step
        history.append(HistoryPoint(step, int(lengths.min()), float(lengths.mean())))
        if config.target_length is not None and best_length <= config.target_length:
            break
    return SolverResult(
        algorithm, best_tour, best_length, best_step,
        tuple(history), config.seed,
    )


def solve_genetic(
    coordinates: np.ndarray, config: GAConfig = GAConfig()
) -> SolverResult:
    return _solve_genetic(coordinates, config, "genetic-algorithm")


run_ga = solve_genetic
