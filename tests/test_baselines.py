import unittest

import numpy as np

from algorithms.nearest_neighbor import solve_nearest_neighbor
from core.tours import validate_tour
from algorithms.two_opt import solve_two_opt


COORDINATES = np.array([
    [0.0, 0.0], [3.0, 0.0], [4.0, 2.0], [3.0, 4.0],
    [0.0, 4.0], [-1.0, 2.0],
])


class BaselineTests(unittest.TestCase):
    def test_nearest_neighbor_is_deterministic_and_valid(self) -> None:
        first = solve_nearest_neighbor(COORDINATES)
        second = solve_nearest_neighbor(COORDINATES)
        np.testing.assert_array_equal(first.best_tour, second.best_tour)
        validate_tour(first.best_tour, len(COORDINATES))

    def test_two_opt_never_worsens_initial_tour(self) -> None:
        nearest = solve_nearest_neighbor(COORDINATES)
        improved = solve_two_opt(COORDINATES, initial_tour=nearest.best_tour)
        self.assertLessEqual(improved.best_length, nearest.best_length)
        validate_tour(improved.best_tour, len(COORDINATES))

    def test_two_opt_history_is_monotonic(self) -> None:
        result = solve_two_opt(COORDINATES)
        lengths = [point.best_length for point in result.history]
        self.assertEqual(lengths, sorted(lengths, reverse=True))


if __name__ == "__main__":
    unittest.main()

