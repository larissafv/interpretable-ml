"""Permutation Feature Importance (PFI) for time series.

Computes feature importance by permuting (or replacing) values for each
variate/timestep (or slice) and measuring the impact on prediction accuracy.
"""

import copy
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

from common.tools import normalize_score
from common.views import plot_timeseries_with_heatmap
from instances.format_instance import FormatInstance
from model.model import Model
from permutation_feature_importance.structs import PFIConfig


class PermutationFeatureImportance:
    """Calculate permutation feature importance for a given model and dataset."""

    def __init__(self, model: Model, config: PFIConfig) -> None:
        """Store model and configuration parameters."""
        self._model = model
        self._simplify_algorithm = config.simplify_algorithm
        self._n_slices_ts = config.n_slices_ts
        self._feat_max = config.feat_max
        self._feat_min = config.feat_min

    def explain(
        self,
        instances: np.ndarray,
        ground_truth: np.ndarray,
        format_instance: FormatInstance,
        variate_labels: list[str],
        class_labels: list[str]
    ) -> None:
        """Compute PFI scores and render heatmap plots.

        If `class_labels` is None, a single plot is generated; otherwise one
        plot per class is produced.
        """
        self._format_instance = format_instance
        results = self._calculate_pfi(instances, ground_truth)
        if class_labels is None:
            plot_name = Path("pfi.png")
            results = [result[0] for result in results]  # Single-label case
            plot_timeseries_with_heatmap(instances[0], results, plot_name, variate_labels)
            return None
        for i, class_label in enumerate(class_labels):
            plot_name = Path(f"pfi_{class_label}.png")
            results_i = [result[i] for result in results]
            plot_timeseries_with_heatmap(instances[0], results_i, plot_name, variate_labels)
        return None

    def _calculate_pfi(
        self,
        instances: np.ndarray,
        ground_truth: np.ndarray
    ) -> list:
        """Return PFI metric per (variate, timestep or slice) and class.

        The algorithm optionally operates on fixed-width slices when
        `self._simplify_algorithm` is True, reducing computation by
        treating each slice as a unit of permutation.
        """
        unformatted_input = self._format_instance.unformat(instances)
        y_pred = self._model.predict(unformatted_input)
        normalized_y_pred = normalize_score(ground_truth, y_pred)

        results = []
        timesteps = instances.shape[2]
        if self._simplify_algorithm:
            slices_width = instances.shape[2] // self._n_slices_ts
            timesteps = self._n_slices_ts

        for variate in range(instances.shape[1]):
            for timestep in range(timesteps):
                idx = timestep * slices_width if self._simplify_algorithm else timestep
                input_instances_max = []
                input_instances_min = []
                for instance in instances:
                    if self._feat_max is None or self._feat_min is None:
                        self._feat_max = (
                            instance[variate].max() + 0.1 * abs(instance[variate].max())
                        )
                        self._feat_min = (
                            instance[variate].min() - 0.1 * abs(instance[variate].min())
                        )

                    new_instance_max = copy.deepcopy(instance)
                    if self._simplify_algorithm:
                        new_instance_max[variate, idx: idx + slices_width] = self._feat_max
                    else:
                        new_instance_max[variate, idx] = self._feat_max
                    input_instances_max.append(new_instance_max)

                    new_instance_min = copy.deepcopy(instance)
                    if self._simplify_algorithm:
                        new_instance_min[variate, idx: idx + slices_width] = self._feat_min
                    else:
                        new_instance_min[variate, idx] = self._feat_min
                    input_instances_min.append(new_instance_min)

                unformatted_input_max = self._format_instance.unformat(
                    np.array(input_instances_max))
                unformatted_input_min = self._format_instance.unformat(
                    np.array(input_instances_min))
                
                y_pred_max = self._model.predict(unformatted_input_max)
                y_pred_min = self._model.predict(unformatted_input_min)

                normalized_y_pred_max = normalize_score(normalized_y_pred, y_pred_max)
                normalized_y_pred_min = normalize_score(normalized_y_pred, y_pred_min)

                if normalized_y_pred.shape[1] > 1:
                    acc_max = [
                        accuracy_score(
                            np.transpose(normalized_y_pred)[k],
                            np.transpose(normalized_y_pred_max)[k]
                        )
                        for k in range(normalized_y_pred.shape[1])
                    ]
                    acc_min = [
                        accuracy_score(
                            np.transpose(normalized_y_pred)[k],
                            np.transpose(normalized_y_pred_min)[k]
                        )
                        for k in range(normalized_y_pred.shape[1])
                    ]
                else:
                    acc_max = [accuracy_score(normalized_y_pred, normalized_y_pred_max)]
                    acc_min = [accuracy_score(normalized_y_pred, normalized_y_pred_min)]
                
                # For multi-label, we have one accuracy per label
                metrics = []
                for k in range(len(acc_max)):
                    metrics.append(min(acc_max[k], acc_min[k]))

                if self._simplify_algorithm:
                    # Expand each result by repeating it slices_width times, then sum element-wise
                    results.extend([metrics] * slices_width)
                else:
                    results.append(metrics)

        return results
