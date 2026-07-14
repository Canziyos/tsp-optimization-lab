import unittest
from pathlib import Path

import numpy as np

from algorithms.genetic import GAConfig, solve_genetic
from core.tours import validate_tour
from datasets.tsplib import load_tsplib


COORDINATES = np.array([
    [0.0, 0.0], [3.0, 0.0], [4.0, 2.0], [3.0, 4.0],
    [0.0, 4.0], [-1.0, 2.0],
])


class GeneticTests(unittest.TestCase):
    def test_run_is_deterministic_and_preserves_valid_tours(self) -> None:
        config = GAConfig(population_size=20, generations=25, elite_count=2, seed=7)
        first = solve_genetic(COORDINATES, config)
        second = solve_genetic(COORDINATES, config)
        self.assertEqual(first.best_length, second.best_length)
        np.testing.assert_array_equal(first.best_tour, second.best_tour)
        validate_tour(first.best_tour, len(COORDINATES))

    def test_target_can_stop_run_early(self) -> None:
        config = GAConfig(
            population_size=20,
            generations=100,
            elite_count=2,
            seed=7,
            target_length=20,
        )
        result = solve_genetic(COORDINATES, config)
        self.assertLess(len(result.history) - 1, config.generations)
        self.assertLessEqual(result.best_length, 20)

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GAConfig(population_size=4, elite_count=4)

    def test_historical_configuration_reaches_feasible_berlin52_tour(self) -> None:
        data = (
            Path(__file__).resolve().parents[1]
            / "datasets"
            / "berlin52.tsp"
        )
        coordinates, _ = load_tsplib(data)
        result = solve_genetic(coordinates, GAConfig(target_length=8000))
        self.assertEqual(result.best_length, 7998)
        self.assertEqual(result.best_step, 300)


if __name__ == "__main__":
    unittest.main()
