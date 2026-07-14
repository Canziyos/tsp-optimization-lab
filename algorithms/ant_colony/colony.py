"""Coordination and evaluation of one ant-colony iteration."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.tours import population_lengths

from .ant import Ant
from .config import ACOConfig
from .pheromone import PheromoneMap


IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ColonyResult:
    """Tours and measurements produced by one colony iteration."""

    tours: IntArray
    lengths: IntArray
    best_index: int

    @property
    def best_tour(self) -> IntArray:
        return self.tours[self.best_index]

    @property
    def best_length(self) -> int:
        return int(self.lengths[self.best_index])

    @property
    def mean_length(self) -> float:
        return float(self.lengths.mean())


def construct_colony_tours(
    pheromone: PheromoneMap,
    heuristic: FloatArray,
    config: ACOConfig,
    rng: np.random.Generator,
) -> IntArray:
    """Let every ant independently construct one closed tour."""
    city_count = pheromone.values.shape[0]
    tours = [
        Ant(city_count=city_count, start_city=0).construct_tour(
            pheromone=pheromone,
            heuristic=heuristic,
            config=config,
            rng=rng,
        )
        for _ in range(config.ant_count)
    ]
    return np.stack(tours)


def evaluate_colony(tours: IntArray, distances: IntArray) -> ColonyResult:
    """Calculate every tour length and identify the best ant."""
    if tours.ndim != 2 or len(tours) == 0:
        raise ValueError("Colony tours must be a non-empty two-dimensional array.")
    lengths = population_lengths(tours, distances).astype(np.int64, copy=False)
    return ColonyResult(tours=tours, lengths=lengths, best_index=int(np.argmin(lengths)))


def run_colony_iteration(
    pheromone: PheromoneMap,
    heuristic: FloatArray,
    distances: IntArray,
    config: ACOConfig,
    rng: np.random.Generator,
) -> ColonyResult:
    """Construct and evaluate all tours for one iteration."""
    return evaluate_colony(
        construct_colony_tours(pheromone, heuristic, config, rng),
        distances,
    )
