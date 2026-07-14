"""Configuration for the 2-opt local-search solver."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TwoOptConfig:
    max_passes: int = 100
    first_improvement: bool = True
    target_length: int | None = None

    def __post_init__(self) -> None:
        if self.max_passes < 1:
            raise ValueError("Maximum pass count must be positive.")

        if self.target_length is not None and self.target_length <= 0:
            raise ValueError("Target length must be positive.")