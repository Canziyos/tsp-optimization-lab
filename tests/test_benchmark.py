import unittest

import numpy as np

from experiments.benchmark import run_benchmark
from algorithms.ant_colony import ACOConfig
from algorithms.genetic import GAConfig
from algorithms.hybrid import HybridConfig


COORDINATES = np.array([
    [0.0, 0.0], [3.0, 0.0], [4.0, 2.0],
    [3.0, 4.0], [0.0, 4.0], [-1.0, 2.0],
])


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_runs_baselines_once_and_stochastic_solvers_per_seed(self) -> None:
        config = GAConfig(population_size=12, generations=5, elite_count=1)
        aco = ACOConfig(ant_count=4, iterations=3, ranked_ants=2)
        hybrid = HybridConfig(genetic=config, refine_every=2, refine_count=2)
        runs = run_benchmark(
            COORDINATES, (3, 4), config, aco, hybrid
        )
        self.assertEqual([run.algorithm for run in runs], [
            "nearest-neighbor", "two-opt", "genetic-algorithm", "genetic-algorithm",
            "ant-colony", "ant-colony",
            "hybrid-genetic-two-opt", "hybrid-genetic-two-opt",
        ])
        self.assertEqual(
            [run.seed for run in runs], [None, None, 3, 4, 3, 4, 3, 4]
        )
        self.assertTrue(all(run.elapsed_seconds >= 0.0 for run in runs))

    def test_benchmark_requires_a_seed(self) -> None:
        with self.assertRaises(ValueError):
            run_benchmark(COORDINATES, seeds=())


if __name__ == "__main__":
    unittest.main()
