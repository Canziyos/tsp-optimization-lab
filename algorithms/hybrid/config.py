"""Configuration for the hybrid TSP solver."""

from dataclasses import dataclass, field

from algorithms.genetic import GAConfig
from algorithms.two_opt import TwoOptConfig


@dataclass(frozen=True, slots=True)
class HybridConfig:
    genetic: GAConfig = field(default_factory=GAConfig)

    two_opt: TwoOptConfig = field(
        default_factory=lambda: TwoOptConfig(
            max_passes=20,
            first_improvement=True,
        )
    )

    refine_every: int = 10
    refine_count: int = 5

    use_nearest_neighbor_seed: bool = True
    target_length: int | None = None

    def __post_init__(self) -> None:
        if self.refine_every < 1:
            raise ValueError("Refinement interval must be positive.")

        if self.refine_count < 1:
            raise ValueError("Refinement count must be positive.")

        if self.refine_count > self.genetic.population_size:
            raise ValueError(
                "Refinement count cannot exceed the population size."
            )

        if self.target_length is not None and self.target_length <= 0:
            raise ValueError("Target length must be positive.")