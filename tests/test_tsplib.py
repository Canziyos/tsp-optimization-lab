import unittest
from pathlib import Path

import numpy as np

from core.tours import tour_length, validate_tour
from core.distances import distance_matrix
from datasets.tsplib import load_tsplib


DATA = (
    Path(__file__).resolve().parents[1] / "datasets" / "berlin52.tsp"
)
OPTIMAL_TOUR = np.array([
    0, 21, 30, 17, 2, 16, 20, 41, 6, 1, 29, 22, 19, 49, 28, 15, 45,
    43, 33, 34, 35, 38, 39, 36, 37, 47, 23, 4, 14, 5, 3, 24, 11, 27,
    26, 25, 46, 12, 13, 51, 10, 50, 32, 42, 9, 8, 7, 40, 18, 44, 31,
    48, 0,
])


class TsplibTests(unittest.TestCase):
    def test_loads_berlin52(self) -> None:
        coordinates, metadata = load_tsplib(DATA)
        self.assertEqual(coordinates.shape, (52, 2))
        self.assertEqual(metadata["EDGE_WEIGHT_TYPE"], "EUC_2D")

    def test_reference_tour_has_known_optimum(self) -> None:
        coordinates, _ = load_tsplib(DATA)
        validate_tour(OPTIMAL_TOUR, len(coordinates))
        self.assertEqual(tour_length(OPTIMAL_TOUR, distance_matrix(coordinates)), 7542)

    def test_rejects_invalid_tour(self) -> None:
        with self.assertRaises(ValueError):
            validate_tour(np.array([0, 1, 1, 0]), 3)


if __name__ == "__main__":
    unittest.main()
