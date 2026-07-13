import unittest

import numpy as np

from berlin52_ga.operators import mutate_inversion, order_crossover, tournament_select
from berlin52_ga.tours import validate_tour


class OperatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.first = np.array([0, 1, 2, 3, 4, 5, 0])
        self.second = np.array([0, 5, 4, 3, 2, 1, 0])

    def test_order_crossover_keeps_valid_permutation(self) -> None:
        child = order_crossover(self.first, self.second, np.random.default_rng(3))
        validate_tour(child, 6)

    def test_inversion_keeps_depot_and_permutation(self) -> None:
        result = mutate_inversion(self.first, 1.0, np.random.default_rng(4))
        validate_tour(result, 6)
        self.assertFalse(np.shares_memory(result, self.first))

    def test_tournament_returns_population_member(self) -> None:
        population = np.stack((self.first, self.second))
        selected = tournament_select(
            population, np.array([10, 20]), 2, 1.0, np.random.default_rng(1)
        )
        np.testing.assert_array_equal(selected, self.first)


if __name__ == "__main__":
    unittest.main()

