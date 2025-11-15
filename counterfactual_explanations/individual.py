"""Representation and operators for GA individuals in counterfactual search.

Each Individual stores a value (candidate instance) and a fitness score.
Operators implement crossover and mutation tailored to time-series data.
"""

from __future__ import annotations

import copy
import logging

import numpy as np

from common.perturbations import perturb_ecg, perturb_general, perturb_with_slices
from instances.format_instance import FormatInstance
from model.model import Model

logger = logging.getLogger(__name__)


class Individual:
    """GA individual holding a candidate counterfactual instance and fitness."""

    def __init__(
        self, value: np.ndarray, fitness: float | None = None
    ) -> None:
        """Initialize an individual with its value and optional fitness."""
        self.value = value
        self.fitness = fitness

    def crossover(
        self,
        other: Individual,
        model: Model,
        prob_change_piece: float,
        label_idx: int,
        max_retries: int,
        threshold: float,
        goal: int,
        format_instance: FormatInstance,
    ) -> bool:
        """Perform crossover with another individual.

        Swaps elements between `self` and `other` with probability
        `prob_change_piece`, accepting offspring only if they preserve
        the current prediction (i.e., not yet flipping to the goal).
        Returns True if a successful crossover occurred.
        """
        success = False
        child1 = copy.deepcopy(self.value)
        child2 = copy.deepcopy(other.value)

        for _ in range(max_retries):
            crossover_mask = np.random.binomial(1, prob_change_piece, size=child1[0].shape)
            for i in range(child1.shape[1]):
                for j in range(child1.shape[2]):
                    if crossover_mask[i, j] == 1:
                        child1[0, i, j], child2[0, i, j] = child2[0, i, j], child1[0, i, j]

            instance1_pred = model.predict(format_instance.unformat(child1))
            pred1 = 1 if instance1_pred[0][label_idx] > threshold else 0
            if pred1 != goal:
                continue

            instance2_pred = model.predict(format_instance.unformat(child2))
            pred2 = 1 if instance2_pred[0][label_idx] > threshold else 0

            if pred2 == pred1:
                self.value = child1
                other.value = child2
                success = True
                break
        return success

    def mutation(
        self,
        instance: np.ndarray,
        model: Model,
        ce_type: str,
        n_slices: int,
        sampling_rate: int,
        max_range: float,
        prob_change_piece: float,
        label_idx: int,
        max_retries: int,
        threshold: float,
        goal: int,
        format_instance: FormatInstance,
        probs: np.ndarray | None = None,
    ) -> bool:
        """Mutate the individual according to the selected CE strategy.

        Returns True if a successful mutation (that maintains current prediction)
        is found and applied within the retry budget.
        """
        success = False
        for _ in range(max_retries):
            if ce_type == "ecg":
                new_instance = perturb_ecg(
                    self.value[0],
                    instance[0],
                    1,
                    sampling_rate,
                    max_range,
                    None,
                    prob_change_piece,
                    probs,
                    False
                )
            elif ce_type == "with_slices":
                slices_width = self.value.shape[1] // n_slices
                new_instance = perturb_with_slices(
                    self.value[0],
                    1,
                    n_slices,
                    slices_width,
                    None,
                    prob_change_piece,
                    probs,
                    False
                )
            elif ce_type == "general":
                new_instance = perturb_general(
                    self.value[0], 1, max_range, None, prob_change_piece, probs, False
                )

            new_instance_pred = model.predict(format_instance.unformat(new_instance))
            pred = 1 if new_instance_pred[0][label_idx] > threshold else 0
            if pred == goal:
                self.value = new_instance
                success = True
                break
        return success
    
    def crossover_original(
        self,
        original: np.ndarray,
        model: Model,
        prob_change_piece: float,
        label_idx: int,
        max_retries: int,
        threshold: float,
        goal: int,
        format_instance: FormatInstance,
    ) -> bool:
        """Perform crossover with another individual.

        Swaps elements between `self` and `other` with probability
        `prob_change_piece`, accepting offspring only if they preserve
        the current prediction (i.e., not yet flipping to the goal).
        Returns True if a successful crossover occurred.
        """
        success = False
        child1 = copy.deepcopy(self.value)

        for _ in range(max_retries):
            crossover_mask = np.random.binomial(1, prob_change_piece, size=child1[0].shape)
            for i in range(child1.shape[1]):
                for j in range(child1.shape[2]):
                    if crossover_mask[i, j] == 1:
                        child1[0, i, j] = original[0, i, j]

            instance1_pred = model.predict(format_instance.unformat(child1))
            pred1 = 1 if instance1_pred[0][label_idx] > threshold else 0
            if pred1 == goal:
                self.value = child1
                success = True
                break
            
        return success