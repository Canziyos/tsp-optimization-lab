"""Configuration for the nearest-neighbor TSP solver."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NearestNeighborConfig:
    start_city: int = 0
    multi_start: bool = False

    def __post_init__(self) -> None:
        if self.start_city < 0:
            raise ValueError("Start city cannot be negative.")