import unittest
from pathlib import Path

import numpy as np

from tsp_optimization_lab.genetic import GAConfig
from tsp_optimization_lab.hybrid import HybridConfig, solve_hybrid
from tsp_optimization_lab.tours import validate_tour
from tsp_optimization_lab.tsplib import load_tsplib


COORDINATES = np.array([
    [0.0, 0.0], [3.0, 0.0], [4.0, 2.0],
    [3.0, 4.0], [0.0, 4.0], [-1.0, 2.0],
])


class HybridTests(unittest.TestCase):
    def test_run_is_deterministic_and_preserves_valid_tours(self) -> None:
        ga = GAConfig(population_size=12, generations=10, elite_count=2, seed=7)
        config = HybridConfig(ga, refine_interval=2, refine_count=3, refine_moves=1)
        first = solve_hybrid(COORDINATES, config)
        second = solve_hybrid(COORDINATES, config)
        validate_tour(first.best_tour, len(COORDINATES))
        np.testing.assert_array_equal(first.best_tour, second.best_tour)

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HybridConfig(refine_count=0)

    def test_default_seed_reaches_berlin52_optimum(self) -> None:
        path = Path(__file__).parents[1] / "src/tsp_optimization_lab/data/berlin52.tsp"
        coordinates, _ = load_tsplib(path)
        result = solve_hybrid(coordinates)
        self.assertEqual(result.best_length, 7542)


if __name__ == "__main__":
    unittest.main()
