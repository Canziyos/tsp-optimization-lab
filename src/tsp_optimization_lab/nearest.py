"""Deterministic nearest-neighbour TSP baseline."""

import numpy as np

from .models import HistoryPoint, SolverResult
from .tours import tour_length
from .tsplib import distance_matrix


def solve_nearest_neighbor(coordinates: np.ndarray) -> SolverResult:
    distances = distance_matrix(coordinates)
    city_count = len(coordinates)
    if city_count < 3:
        raise ValueError("At least three cities are required.")

    unvisited = set(range(1, city_count))
    route = [0]
    while unvisited:
        current = route[-1]
        next_city = min(unvisited, key=lambda city: (distances[current, city], city))
        route.append(next_city)
        unvisited.remove(next_city)
    route.append(0)
    tour = np.asarray(route, dtype=np.int64)
    length = tour_length(tour, distances)
    history = (HistoryPoint(0, length),)
    return SolverResult("nearest-neighbor", tour, length, 0, history)

