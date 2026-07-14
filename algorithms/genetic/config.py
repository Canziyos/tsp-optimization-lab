"""Configuration for the genetic TSP solver."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GAConfig:
    population_size: int = 200
    generations: int = 1000

    crossover_probability: float = 0.80
    mutation_probability: float = 0.05

    tournament_size: int = 4
    tournament_win_probability: float = 0.75

    elite_count: int = 10

    seed: int = 2
    target_length: int | None = None
    progress_every: int = 0

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("Population size must be at least two.")

        if self.generations < 1:
            raise ValueError("Generation count must be positive.")

        if not 0 <= self.elite_count < self.population_size:
            raise ValueError(
                "Elite count must be non-negative and smaller "
                "than the population size."
            )

        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError(
                "Tournament size must be between one and "
                "the population size."
            )

        probabilities = {
            "Crossover probability": self.crossover_probability,
            "Mutation probability": self.mutation_probability,
            "Tournament win probability": self.tournament_win_probability,
        }

        for name, value in probabilities.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one.")

        if self.target_length is not None and self.target_length <= 0:
            raise ValueError("Target length must be positive.")

        if self.progress_every < 0:
            raise ValueError("Progress interval cannot be negative.")
