"""Minimal parser and EUC_2D distance support for TSPLIB coordinate files."""

from pathlib import Path

import numpy as np


def load_tsplib(path: str | Path) -> tuple[np.ndarray, dict[str, str]]:
    metadata: dict[str, str] = {}
    coordinates: list[tuple[int, float, float]] = []
    in_coordinates = False

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line == "EOF":
            continue
        if line == "NODE_COORD_SECTION":
            in_coordinates = True
            continue
        if in_coordinates:
            node, x, y = line.split()
            coordinates.append((int(node), float(x), float(y)))
        elif ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()

    if not coordinates:
        raise ValueError(f"No NODE_COORD_SECTION found in {path}.")
    coordinates.sort(key=lambda row: row[0])
    expected_ids = list(range(1, len(coordinates) + 1))
    if [row[0] for row in coordinates] != expected_ids:
        raise ValueError("Node identifiers must be contiguous and one-based.")
    if metadata.get("EDGE_WEIGHT_TYPE") != "EUC_2D":
        raise ValueError("Only TSPLIB EUC_2D instances are supported.")

    points = np.asarray([(x, y) for _, x, y in coordinates], dtype=np.float64)
    dimension = int(metadata.get("DIMENSION", len(points)))
    if len(points) != dimension:
        raise ValueError(f"Expected {dimension} nodes, found {len(points)}.")
    return points, metadata


def distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Coordinates must have shape (n_cities, 2).")
    deltas = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    return np.floor(distances + 0.5).astype(np.int64)

