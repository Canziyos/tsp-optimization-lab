"""Configuration for ant-colony optimization."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ACOConfig:
    ant_count: int = 52
    iterations: int = 200
    alpha: float = 1.0
    beta: float = 3.0
    evaporation: float = 0.10
    deposit_weight: float = 100.0
    ranked_ants: int = 10
    elite_weight: float = 0.0
    seed: int = 2
    target_length: int | None = None
    progress_every: int = 0

    def __post_init__(self) -> None:
        if self.ant_count < 1:
            raise ValueError("Ant count must be positive.")

        if self.iterations < 1:
            raise ValueError("Iteration count must be positive.")

        if self.ranked_ants < 1:
            raise ValueError("At least one ranked ant is required.")

        if self.alpha < 0.0 or self.beta < 0.0:
            raise ValueError("Alpha and beta must be non-negative.")

        if self.alpha == 0.0 and self.beta == 0.0:
            raise ValueError("Alpha and beta cannot both be zero.")

        if not 0.0 < self.evaporation < 1.0:
            raise ValueError("Evaporation must be between zero and one.")

        if self.deposit_weight <= 0.0:
            raise ValueError("Deposit weight must be positive.")

        if self.elite_weight < 0.0:
            raise ValueError("Elite weight cannot be negative.")

        if self.target_length is not None and self.target_length <= 0:
            raise ValueError("Target length must be positive.")

        if self.progress_every < 0:
            raise ValueError("Progress interval cannot be negative.")
