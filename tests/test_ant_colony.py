import unittest
from pathlib import Path

import numpy as np

from algorithms.ant_colony import ACOConfig, solve_ant_colony
from core.tours import validate_tour
from datasets.tsplib import load_tsplib


COORDINATES = np.array([
    [0.0, 0.0], [3.0, 0.0], [4.0, 2.0],
    [3.0, 4.0], [0.0, 4.0], [-1.0, 2.0],
])


class AntColonyTests(unittest.TestCase):
    def test_run_is_deterministic_and_preserves_valid_tours(self) -> None:
        config = ACOConfig(ant_count=8, iterations=10, seed=7)
        first = solve_ant_colony(COORDINATES, config)
        second = solve_ant_colony(COORDINATES, config)
        validate_tour(first.best_tour, len(COORDINATES))
        np.testing.assert_array_equal(first.best_tour, second.best_tour)
        self.assertEqual(first.best_length, second.best_length)

    def test_best_history_is_monotonic(self) -> None:
        result = solve_ant_colony(
            COORDINATES, ACOConfig(ant_count=6, iterations=8)
        )
        best = [point.best_length for point in result.history]
        self.assertEqual(best, sorted(best, reverse=True))

    def test_target_can_stop_run_early(self) -> None:
        result = solve_ant_colony(
            COORDINATES, ACOConfig(ant_count=8, iterations=20, target_length=20)
        )
        self.assertLess(len(result.history), 20)

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ACOConfig(evaporation=1.0)

    def test_default_seed_reaches_competitive_berlin52_tour(self) -> None:
        path = Path(__file__).parents[1] / "datasets/berlin52.tsp"
        coordinates, _ = load_tsplib(path)
        result = solve_ant_colony(coordinates)
        self.assertLessEqual(result.best_length, 7800)


if __name__ == "__main__":
    unittest.main()
