"""Distance-related utilities for TSP instances."""

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def distance_matrix(coordinates: FloatArray) -> IntArray:
    """
    Build a symmetric Euclidean distance matrix.

    Distances are rounded to the nearest integer, matching common
    TSPLIB-style Euclidean instances.
    """
    if coordinates.ndim != 2:
        raise ValueError("Coordinates must be a two-dimensional array.")

    if coordinates.shape[1] != 2:
        raise ValueError("Each city must contain exactly two coordinates.")

    if len(coordinates) < 2:
        raise ValueError("At least two cities are required.")

    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Coordinates contain non-finite values.")

    differences = (
        coordinates[:, np.newaxis, :]
        - coordinates[np.newaxis, :, :]
    )

    distances = np.sqrt(
        np.sum(differences**2, axis=2)
    )

    matrix = np.rint(distances).astype(np.int64)

    np.fill_diagonal(matrix, 0)

    return matrix