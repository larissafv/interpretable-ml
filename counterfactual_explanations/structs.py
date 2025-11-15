"""Configuration structures for counterfactual explanations.

Defines dataclasses that capture hyperparameters and required settings for
the genetic algorithm-based counterfactual explanation method.
"""

import dataclasses as dc

import json2dc


@json2dc.add_from_json
@dc.dataclass(kw_only=True)
class CEConfig(json2dc.AddFromJsonInterface):
    """Counterfactual explanation configuration.

    Attributes:
        ce_type: One of {"ecg", "with_slices", "general"} selecting perturbations.
        n_slices: Number of slices for the "with_slices" strategy.
        sampling_rate: ECG sampling rate required for "ecg" strategy.
        max_range: Max random offset range for "general" strategy.
        pop_size: Population size for genetic algorithm.
        n_generations: Number of generations to evolve.
        min_fitness: Early stopping target; stop when best fitness <= min_fitness.
        prob_crossover: Probability of performing crossover.
        prob_mutation: Probability of performing mutation per child.
        prob_change_piece: Per-piece probability used inside operators.
        tournament_size: Tournament size for parent selection.
        label_idx: Target label index to flip/change prediction.
        max_retries: Max attempts for successful genetic operations.
    """
    ce_type: str
    n_slices: int | None
    sampling_rate: int | None
    max_range: float | None
    pop_size: int
    n_generations: int
    min_fitness: float | None
    prob_crossover: float
    prob_mutation: float
    prob_change_piece: float
    tournament_size: int
    label_idx: int
    max_retries: int

    def __post_init__(self) -> None:
        if self.ce_type not in ["ecg", "with_slices", "general"]:
            raise ValueError(f"Counterfactual explanation type {self.ce_type} is not supported.")
        if self.n_slices is None and self.ce_type == "with_slices":
            raise ValueError(
                "n_slices must be specified for with_slices counterfactual explanation type."
            )
        if self.sampling_rate is None and self.ce_type == "ecg":
            raise ValueError(
                "sampling_rate must be specified for ECG counterfactual explanation type."
            )
        if self.max_range is None and self.ce_type == "general":
            raise ValueError(
                "max_range must be specified for general counterfactual explanation type."
            )
        if self.pop_size <= 0:
            raise ValueError("pop_size must be a positive integer.")
        if self.n_generations <= 0:
            raise ValueError("n_generations must be a positive integer.")
        if not (0 <= self.prob_crossover <= 1):
            raise ValueError("prob_crossover must be between 0 and 1.")
        if not (0 <= self.prob_mutation <= 1):
            raise ValueError("prob_mutation must be between 0 and 1.")
        if self.tournament_size <= 0:
            raise ValueError("tournament_size must be a positive integer.")
