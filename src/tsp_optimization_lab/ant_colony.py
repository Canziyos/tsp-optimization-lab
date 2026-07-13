"""Seeded ant-colony optimization for symmetric TSP instances."""

from dataclasses import dataclass

import numpy as np

from .models import HistoryPoint, SolverResult
from .tours import population_lengths
from .tsplib import distance_matrix


@dataclass(frozen=True, slots=True)
class ACOConfig:
    ant_count: int = 52
    iterations: int = 200
    alpha: float = 1.0
    beta: float = 3.0
    evaporation: float = 0.10
    deposit_weight: float = 100.0
    ranked_ants: int = 10
    elite_weight: float = 0.0
    seed: int = 2
    target_length: int | None = None

    def __post_init__(self) -> None:
        if self.ant_count < 1 or self.iterations < 1:
            raise ValueError("Ant count and iterations must be positive.")
        if self.ranked_ants < 1:
            raise ValueError("At least one ranked ant is required.")
        if self.alpha < 0.0 or self.beta < 0.0:
            raise ValueError("Alpha and beta must be non-negative.")
        if not 0.0 < self.evaporation < 1.0:
            raise ValueError("Evaporation must be between zero and one.")
        if self.deposit_weight <= 0.0 or self.elite_weight < 0.0:
            raise ValueError("Deposit weights must be positive or zero for elite_weight.")


def _construct_tour(
    pheromone: np.ndarray,
    heuristic: np.ndarray,
    config: ACOConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    city_count = len(pheromone)
    tour = np.empty(city_count + 1, dtype=np.int64)
    tour[0] = tour[-1] = 0
    unvisited = np.ones(city_count, dtype=bool)
    unvisited[0] = False
    current = 0
    for index in range(1, city_count):
        candidates = np.flatnonzero(unvisited)
        weights = (
            pheromone[current, candidates] ** config.alpha
            * heuristic[current, candidates] ** config.beta
        )
        total = float(weights.sum())
        probabilities = weights / total if total > 0.0 else None
        current = int(rng.choice(candidates, p=probabilities))
        tour[index] = current
        unvisited[current] = False
    return tour


def _deposit(
    pheromone: np.ndarray, tours: np.ndarray, amounts: np.ndarray
) -> None:
    for tour, amount in zip(tours, amounts):
        first, second = tour[:-1], tour[1:]
        np.add.at(pheromone, (first, second), amount)
        np.add.at(pheromone, (second, first), amount)


def solve_ant_colony(
    coordinates: np.ndarray, config: ACOConfig = ACOConfig()
) -> SolverResult:
    distances = distance_matrix(coordinates)
    city_count = len(distances)
    if city_count < 3:
        raise ValueError("At least three cities are required.")
    heuristic = np.zeros_like(distances, dtype=float)
    np.divide(1.0, distances, out=heuristic, where=distances > 0)
    pheromone = np.ones_like(heuristic)
    rng = np.random.default_rng(config.seed)
    best_tour, best_length, best_step = None, np.iinfo(np.int64).max, 0
    history: list[HistoryPoint] = []

    for step in range(1, config.iterations + 1):
        tours = np.stack([
            _construct_tour(pheromone, heuristic, config, rng)
            for _ in range(config.ant_count)
        ])
        lengths = population_lengths(tours, distances)
        current = int(np.argmin(lengths))
        if lengths[current] < best_length:
            best_tour = tours[current].copy()
            best_length, best_step = int(lengths[current]), step
        history.append(HistoryPoint(step, int(best_length), float(lengths.mean())))
        pheromone *= 1.0 - config.evaporation
        ranked = np.argsort(lengths)[: min(config.ranked_ants, config.ant_count)]
        rank_weights = np.arange(len(ranked), 0, -1, dtype=float)
        _deposit(
            pheromone, tours[ranked],
            config.deposit_weight * rank_weights / lengths[ranked],
        )
        _deposit(
            pheromone, best_tour[None, :],
            np.array([config.elite_weight * config.deposit_weight / best_length]),
        )
        if config.target_length is not None and best_length <= config.target_length:
            break
    return SolverResult(
        "ant-colony", best_tour, int(best_length), best_step,
        tuple(history), config.seed,
    )
