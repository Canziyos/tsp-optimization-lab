"""Selection, order crossover, and inversion mutation operators."""

import numpy as np


def tournament_select(
    population: np.ndarray,
    lengths: np.ndarray,
    size: int,
    win_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if not 1 <= size <= len(population):
        raise ValueError("Tournament size must be within the population.")
    if not 0.0 <= win_probability <= 1.0:
        raise ValueError("Tournament win probability must be in [0, 1].")

    candidates = rng.choice(len(population), size=size, replace=False)
    ranked = candidates[np.argsort(lengths[candidates])]
    if len(ranked) == 1 or rng.random() < win_probability:
        winner = ranked[0]
    else:
        winner = rng.choice(ranked[1:])
    return population[winner].copy()


def order_crossover(
    first: np.ndarray, second: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    if first.shape != second.shape or len(first) < 4:
        raise ValueError("Parents must be equal closed tours with at least three cities.")
    left, right = sorted(rng.choice(np.arange(1, len(first) - 1), 2, replace=False))
    child = np.full_like(first, -1)
    child[0] = child[-1] = 0
    child[left : right + 1] = first[left : right + 1]

    used = set(child[left : right + 1].tolist())
    interior_count = len(child) - 2
    wrapped = [1 + ((right + offset - 1) % interior_count)
               for offset in range(1, interior_count + 1)]
    fill_values = [second[index] for index in wrapped if second[index] not in used]
    fill_positions = [index for index in wrapped if child[index] == -1]
    child[fill_positions] = fill_values
    return child


def mutate_inversion(
    tour: np.ndarray, probability: float, rng: np.random.Generator
) -> np.ndarray:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Mutation probability must be in [0, 1].")
    mutated = tour.copy()
    if rng.random() < probability:
        left, right = sorted(
            rng.choice(np.arange(1, len(tour) - 1), 2, replace=False)
        )
        mutated[left : right + 1] = mutated[left : right + 1][::-1]
    return mutated
