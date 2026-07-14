"""Public interface for nearest-neighbor TSP search."""

from .config import NearestNeighborConfig
from .solver import (
    build_nearest_neighbor_tour,
    solve_nearest_neighbor,
    normalize_tour_start,
)

__all__ = [
    "NearestNeighborConfig",
    "build_nearest_neighbor_tour",
    "solve_nearest_neighbor",
    "normalize_tour_start",
]