"""Tour representation, validation, creation, and evaluation."""

import numpy as np


def validate_tour(tour: np.ndarray, city_count: int) -> None:
    candidate = np.asarray(tour)
    if candidate.ndim != 1 or len(candidate) != city_count + 1:
        raise ValueError("A closed tour must contain n_cities + 1 entries.")
    if candidate[0] != 0 or candidate[-1] != 0:
        raise ValueError("Tours must start and end at city 0.")
    if sorted(candidate[1:-1].tolist()) != list(range(1, city_count)):
        raise ValueError("Every non-depot city must appear exactly once.")


def random_tour(city_count: int, rng: np.random.Generator) -> np.ndarray:
    if city_count < 3:
        raise ValueError("At least three cities are required.")
    interior = rng.permutation(np.arange(1, city_count, dtype=np.int64))
    return np.concatenate(([0], interior, [0]))


def initial_population(
    size: int, city_count: int, rng: np.random.Generator
) -> np.ndarray:
    if size < 2:
        raise ValueError("Population size must be at least two.")
    return np.stack([random_tour(city_count, rng) for _ in range(size)])


def tour_length(tour: np.ndarray, distances: np.ndarray) -> int:
    candidate = np.asarray(tour, dtype=np.int64)
    return int(distances[candidate[:-1], candidate[1:]].sum())


def population_lengths(population: np.ndarray, distances: np.ndarray) -> np.ndarray:
    return np.asarray([tour_length(tour, distances) for tour in population])

