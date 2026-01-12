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
        self, value: np.ndarray, prediction: float | None = None, fitness: float | None = None
    ) -> None:
        """Initialize an individual with its value, prediction and fitness."""
        self.value = value
        self.prediction = prediction
        self.fitness = fitness

    def crossover(
        self,
        other: Individual,
        model: Model,
        prob_change_piece: float,
        label_idx: int,
        max_retries: int,
        original_pred: int,
        sign: bool,
        format_instance: FormatInstance,
    ) -> bool:
        """Perform crossover with another individual.

        Swaps elements between `self` and `other` with probability
        `prob_change_piece`, accepting offspring only if they change
        the current prediction.
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
            pred_label1 = instance1_pred[0][label_idx]

            # If the prediction did not change in the desired direction, skip
            if (not (sign and pred_label1 < original_pred) and
                    not (not sign and pred_label1 > original_pred)):
                continue

            #pred1 = 1 if instance1_pred[0][label_idx] > threshold else 0
            #if pred1 != goal:
            #    continue

            instance2_pred = model.predict(format_instance.unformat(child2))
            pred_label2 = instance2_pred[0][label_idx]

            # If the prediction was True, then the new prediction must be lower than original
            if sign and pred_label2 < original_pred:
                self.value = child1
                self.prediction = pred_label1
                other.value = child2
                other.prediction = pred_label2
                success = True
                break

            # If the prediction was False, then the new prediction must be higher than original
            elif not sign and pred_label2 > original_pred:
                self.value = child1
                self.prediction = pred_label1
                other.value = child2
                other.prediction = pred_label2
                success = True
                break

            #pred2 = 1 if instance2_pred[0][label_idx] > threshold else 0

            #if pred2 == pred1:
            #    self.value = child1
            #    other.value = child2
            #    success = True
            #    break
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
        original_pred: int,
        sign: bool,
        format_instance: FormatInstance,
        probs: np.ndarray | None = None,
    ) -> bool:
        """Mutate the individual according to the selected CE strategy.

        Returns True if a successful mutation (that changes current prediction)
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
            pred_label = new_instance_pred[0][label_idx]

            # If the prediction was True, then the new prediction must be lower than original
            if sign and pred_label < original_pred:
                self.value = new_instance
                self.prediction = pred_label
                success = True
                break

            # If the prediction was False, then the new prediction must be higher than original
            elif not sign and pred_label > original_pred:
                self.value = new_instance
                self.prediction = pred_label
                success = True
                break

            #pred = 1 if new_instance_pred[0][label_idx] > threshold else 0
            #if pred == goal:
            #    self.value = new_instance
            #    success = True
            #    break
        return success
    
    def crossover_original(
        self,
        original: np.ndarray,
        model: Model,
        prob_change_piece: float,
        label_idx: int,
        max_retries: int,
        original_pred: int,
        sign: bool,
        format_instance: FormatInstance,
    ) -> bool:
        """Perform crossover with another individual.

        Swaps elements between `self` and `original` with probability
        `prob_change_piece`, accepting offspring only if they change
        the current prediction.
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
            pred_label1 = instance1_pred[0][label_idx]

            # If the prediction was True, then the new prediction must be lower than original
            if sign and pred_label1 < original_pred:
                self.value = child1
                self.prediction = pred_label1
                success = True
                break

            # If the prediction was False, then the new prediction must be higher than original
            elif not sign and pred_label1 > original_pred:
                self.value = child1
                self.prediction = pred_label1
                success = True
                break


            #pred1 = 1 if instance1_pred[0][label_idx] > threshold else 0
            #if pred1 == goal:
            #    self.value = child1
            #    success = True
            #    break
            
        return success