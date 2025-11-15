from pathlib import Path

import numpy as np

from common.views import plot_two_timeseries_with_heatmap
from instances.format_instance import FormatInstance
from model.model import Model
from ppi.structs import PFILocalConfig


class PFILocal:
    def __init__(self, model: Model, config: PFILocalConfig) -> None:
        self._model = model
        self._batch_size = model._batch_size if model._batch_size % 2 == 0 else model._batch_size + 1  # noqa: E501
        self._max_range = config.max_range

    def explain(
        self,
        instance: np.ndarray,
        format_instance: FormatInstance,
        variate_labels: list[str],
        class_labels: list[str],
        counterfactual_with_pfi: bool
    ) -> None:
        self._format_instance = format_instance
        results, perturbated_instance = self._calculate_pfi_local(instance)
        if class_labels is None:
            plot_name = Path("pfi_local.png")
            results = [result[0] for result in results]  # Single-label case
            plot_two_timeseries_with_heatmap(
                instance[0], perturbated_instance[0], results[0], plot_name, variate_labels
            )
            if counterfactual_with_pfi:
                return results
            return None
        for i, class_label in enumerate(class_labels):
            plot_name = Path(f"pfi_local_{class_label}.png")
            plot_two_timeseries_with_heatmap(
                instance[0], perturbated_instance[0], results[i][0], plot_name, variate_labels
            )
        if counterfactual_with_pfi:
            return results
        return None

    def _calculate_pfi_local(
        self,
        instance: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Compute local permutation feature importance (streamed batches).

        This implementation generates +/− perturbations pair-by-pair and
        predicts them in chunks to avoid storing all perturbed copies in memory
        at once. It preserves the original semantics (choose the perturbation
        direction that produces the largest absolute change) while bounding
        peak memory usage.
        """
        perturbated_instance = np.zeros(instance.shape)

        # baseline prediction
        unformatted_input = self._format_instance.unformat(instance)
        y_pred = self._model.predict(unformatted_input)

        v = instance.shape[1]
        t = instance.shape[2]
        num_outputs = int(y_pred.shape[-1])

        results = [np.zeros(instance.shape) for _ in range(num_outputs)]

        baseline = instance[0]
        batch_samples: list[np.ndarray] = []
        # each entry corresponds to a (+,-) pair; we store the pair position once
        batch_pair_positions: list[tuple[int, int]] = []
        batch_size = max(2, self._batch_size)

        def _flush_batch() -> None:
            if not batch_samples:
                return
            batch_arr = np.array(batch_samples)
            unformatted_batch = self._format_instance.unformat(batch_arr)
            preds = self._model.predict(unformatted_batch)

            # preds has shape (len(batch_arr), num_outputs) with order [+, -, +, -,...]
            for i in range(0, preds.shape[0], 2):
                pair_idx = i // 2
                variate_pos, timestep_pos = batch_pair_positions[pair_idx]
                preds_max = preds[i]
                preds_min = preds[i + 1]

                for out_idx in range(num_outputs):
                    diff_max = preds_max[out_idx] - y_pred[0, out_idx]
                    diff_min = preds_min[out_idx] - y_pred[0, out_idx]

                    if abs(diff_max) > abs(diff_min):
                        perturbated_instance[0, variate_pos, timestep_pos] = (
                            baseline[variate_pos, timestep_pos] + self._max_range
                        )
                        results[out_idx][0][variate_pos][timestep_pos] = diff_max
                    else:
                        perturbated_instance[0, variate_pos, timestep_pos] = (
                            baseline[variate_pos, timestep_pos] - self._max_range
                        )
                        results[out_idx][0][variate_pos][timestep_pos] = diff_min

        # Generate perturbations and process in streaming batches
        for variate in range(v):
            for timestep in range(t):
                plus = baseline.copy()
                plus[variate, timestep] += self._max_range
                batch_samples.append(plus)
                batch_pair_positions.append((variate, timestep))

                minus = baseline.copy()
                minus[variate, timestep] -= self._max_range
                batch_samples.append(minus)

                # When we reach the batch size, predict and flush
                if len(batch_samples) >= batch_size:
                    _flush_batch()
                    batch_samples = []
                    batch_pair_positions = []

        # Flush any remaining
        if batch_samples:
            _flush_batch()

        return results, perturbated_instance
