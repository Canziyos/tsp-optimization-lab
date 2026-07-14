"""Tour construction, validation, and evaluation utilities."""

import numpy as np
from numpy.typing import NDArray


IntArray = NDArray[np.int64]


def validate_tour(
    tour: IntArray,
    city_count: int,
) -> None:
    """Validate a closed tour that starts and ends at city zero."""
    if tour.ndim != 1:
        raise ValueError("Tour must be one-dimensional.")

    if city_count < 2:
        raise ValueError("At least two cities are required.")

    expected_length = city_count + 1

    if len(tour) != expected_length:
        raise ValueError(
            f"Closed tour must contain {expected_length} entries."
        )

    if tour[0] != 0 or tour[-1] != 0:
        raise ValueError("Tour must start and end at city zero.")

    interior = tour[:-1]

    if np.any(interior < 0) or np.any(interior >= city_count):
        raise ValueError("Tour contains invalid city indices.")

    if len(np.unique(interior)) != city_count:
        raise ValueError("Tour must visit every city exactly once.")


def random_tour(
    city_count: int,
    rng: np.random.Generator,
) -> IntArray:
    """Create one random closed tour with city zero fixed as depot."""
    if city_count < 2:
        raise ValueError("At least two cities are required.")

    interior = rng.permutation(
        np.arange(1, city_count, dtype=np.int64)
    )

    return np.concatenate(
        (
            np.array([0], dtype=np.int64),
            interior,
            np.array([0], dtype=np.int64),
        )
    )


def initial_population(
    size: int,
    city_count: int,
    rng: np.random.Generator,
) -> IntArray:
    """Create a population of random closed tours."""
    if size < 1:
        raise ValueError("Population size must be positive.")

    tours = [
        random_tour(
            city_count=city_count,
            rng=rng,
        )
        for _ in range(size)
    ]

    return np.stack(tours)


def tour_length(
    tour: IntArray,
    distances: IntArray,
) -> int:
    """Calculate the total length of one closed tour."""
    city_count = distances.shape[0]

    _validate_distance_matrix(distances)
    validate_tour(tour, city_count)

    from_cities = tour[:-1]
    to_cities = tour[1:]

    return int(
        distances[from_cities, to_cities].sum()
    )


def population_lengths(
    population: IntArray,
    distances: IntArray,
) -> IntArray:
    """Calculate lengths for a population of closed tours."""
    _validate_distance_matrix(distances)

    if population.ndim != 2:
        raise ValueError("Population must be a two-dimensional array.")

    if len(population) == 0:
        raise ValueError("Population cannot be empty.")

    city_count = distances.shape[0]

    if population.shape[1] != city_count + 1:
        raise ValueError(
            "Every closed tour must contain city_count + 1 entries."
        )

    if np.any(population[:, 0] != 0):
        raise ValueError("Every tour must start at city zero.")

    if np.any(population[:, -1] != 0):
        raise ValueError("Every tour must end at city zero.")

    from_cities = population[:, :-1]
    to_cities = population[:, 1:]

    lengths = distances[from_cities, to_cities].sum(axis=1)

    return lengths.astype(np.int64, copy=False)


def _validate_distance_matrix(
    distances: IntArray,
) -> None:
    """Validate a symmetric square distance matrix."""
    if distances.ndim != 2:
        raise ValueError("Distance matrix must be two-dimensional.")

    row_count, column_count = distances.shape

    if row_count != column_count:
        raise ValueError("Distance matrix must be square.")

    if row_count < 2:
        raise ValueError("At least two cities are required.")

    if np.any(distances < 0):
        raise ValueError("Distances cannot be negative.")

    if not np.all(np.isfinite(distances)):
        raise ValueError("Distance matrix contains non-finite values.")

    if not np.array_equal(distances, distances.T):
        raise ValueError("Distance matrix must be symmetric.")

    if np.any(np.diag(distances) != 0):
        raise ValueError("Distance matrix diagonal must be zero.")