"""Perturbation utilities for generating synthetic samples.

This module contains domain-specific and generic perturbation functions used by
explainability methods (LIME, counterfactuals) to create local neighborhoods
around an instance. Functions operate on multivariate time series arrays with
shape (n_variates, n_timesteps).
"""

import copy
import random

import neurokit2 as nk
import numpy as np

from common.views import plot_timeseries_differences, plot_timeseries_with_highlights


def perturb_ecg(
    instance: np.ndarray,
    original: np.ndarray | None,
    n_samples: int,
    sampling_rate: int,
    max_range: float,
    variate_labels: list[str] | None,
    prob: float = 0.35,
    probs: np.ndarray | None = None,
    flag: bool = True
) -> np.ndarray:
    """Generate ECG-specific perturbations based on delineated wave landmarks.

    Uses NeuroKit2 to detect ECG waves (P, QRS, T) and applies randomized
    shifts or mean-replacements to simulate plausible perturbations.

    Args:
        instance: Base instance array of shape (n_leads, n_timesteps).
        n_samples: Number of perturbed samples to generate.
        sampling_rate: ECG sampling rate used by NeuroKit2 for delineation.
        max_range: Maximum offset range when perturbing peak values.
        variate_labels: Optional labels for leads; used for plots when flag=True.
        prob: Probability of perturbing each detected wave/peak set.
        flag: If True, saves one pair of example plots and then disables plotting.

    Returns:
        np.ndarray: Array with shape (n_samples, n_leads, n_timesteps).
    """
    def perturb_peaks(
        lead_sample: np.ndarray,
        key1: str,
        key2: str,
        idx: int
    ) -> None:
        if np.isnan(waves[key1][idx]):
            peak = None
        else:
            peak = int(waves[key1][idx])
        if idx > len(waves[key2]) - 1:
            if np.isnan(waves[key2][-1]):
                peak_ref = None
            else:
                peak_ref = int(waves[key2][-1])
        else:
            if np.isnan(waves[key2][idx]):
                peak_ref = None
            else:
                peak_ref = int(waves[key2][idx])
        if peak is None or peak_ref is None:
            base_diff = None
        else:
            base_diff = abs(lead_sample[peak] - lead_sample[peak_ref])

        # Decide perturbation magnitude
        if max_range is not None:
            diff = float(max_range)
        elif base_diff is not None and base_diff > 0:
            diff = float(base_diff)
        else:
            # Fallback to a small fraction of the signal range
            rng = float(np.ptp(lead_sample)) if lead_sample.size else 1.0
            diff = max(1e-6, 0.05 * rng)

        offset = random.uniform(-diff, diff)  # noqa: S311
        operation = random.choice(["+", "-"])  # noqa: S311
        if operation == "-":
            lead_sample[peak] -= offset
        elif operation == "+":
            lead_sample[peak] += offset

    def perturb_waves(
        lead_sample: np.ndarray,
        start_indices: list[int],
        end_indices: list[int]
    ) -> None:
        # Replace each detected wave segment by its mean value
        n = lead_sample.shape[0]
        for start_idx, end_idx in zip(start_indices, end_indices, strict=False):
            # Sanitize indices (skip NaN/None, clamp to valid bounds, require start < end)
            if start_idx is None or end_idx is None:
                continue
            if not (np.isfinite(start_idx) and np.isfinite(end_idx)):
                continue
            s = int(start_idx)
            e = int(end_idx)
            if s < 0:
                s = 0
            if e > n:
                e = n
            if e <= s:
                continue
            segment = lead_sample[s:e]
            if segment.size == 0:
                continue

            operation = random.choice(["offset", "mean"])  # noqa: S311
            if operation == "mean":
                lead_sample[s:e] = np.mean(segment)
            elif operation == "offset":
                offset = random.uniform(-0.5, 0.5)  # noqa: S311
                operation = random.choice(["+", "-"])  # noqa: S311
                if operation == "-":
                    lead_sample[s:e] -= offset
                elif operation == "+":
                    lead_sample[s:e] += offset

    if original is None:
        original = instance
    samples = []
    for _ in range(n_samples):
        ecg = []

        # Iterate over each lead (row) in the instance
        for i, lead in enumerate(original):
            # Detect R-peaks and delineate ECG waves for the current lead
            try:
                _, rpeaks = nk.ecg_peaks(lead, sampling_rate=sampling_rate)
                _, waves = nk.ecg_delineate(
                    lead, rpeaks, sampling_rate=sampling_rate, method="dwt"
                )
                waves["ECG_R_Peaks"] = rpeaks.get("ECG_R_Peaks", [])
            except Exception:
                # If delineation fails for this lead, keep it unchanged
                ecg.append(copy.deepcopy(instance[i]))
                continue
            
            lead_sample = copy.deepcopy(instance[i])
            # Iterate over each wave type and apply perturbations based on probability
            for key, values in waves.items():
                if not key.endswith("Peaks"):
                    continue

                # Special handling for P and T peaks: perturb entire waves instead
                if key in ["ECG_P_Peaks", "ECG_T_Peaks"]:
                    if probs is not None:
                        prob_list = []
                        for value_idx in values:
                            if np.isnan(value_idx):
                                prob_list.append(0.0)
                            else:
                                prob_list.append(abs(probs[i, int(value_idx)]))
                            perturbation_values = np.random.binomial(1, max(prob_list))
                    else:
                        perturbation_values = np.random.binomial(1, prob)
                    
                    if perturbation_values == 1:
                        if key == "ECG_P_Peaks":
                            perturb_waves(
                                lead_sample,
                                waves.get("ECG_P_Onsets", []),
                                waves.get("ECG_P_Offsets", []),
                            )
                        elif key == "ECG_T_Peaks":
                            perturb_waves(
                                lead_sample,
                                waves.get("ECG_T_Onsets", []),
                                waves.get("ECG_T_Offsets", []),
                            )
                    continue

                # Determine which peaks to perturb
                if probs is not None:
                    prob_list = []
                    for value_idx in values:
                        if np.isnan(value_idx):
                            prob_list.append(0.0)
                        else:
                            prob_list.append(abs(probs[i, int(value_idx)]))
                    perturbation_values = np.random.binomial(
                        1, np.array(prob_list), size=len(values)
                        )
                else:
                    perturbation_values = np.random.binomial(1, prob, size=len(values))

                # Apply perturbations to selected peaks
                for j in range(len(values)):
                    if perturbation_values[j] == 1:
                        if key == "ECG_R_Peaks":
                            perturb_peaks(lead_sample, "ECG_R_Peaks", "ECG_P_Peaks", j)
                        elif key == "ECG_Q_Peaks":
                            perturb_peaks(lead_sample, "ECG_Q_Peaks", "ECG_P_Offsets", j)
                        elif key == "ECG_S_Peaks":
                            perturb_peaks(lead_sample, "ECG_S_Peaks", "ECG_T_Onsets", j)
            ecg.append(lead_sample)
        ecg_arr = np.array(ecg)
        if flag:
            # Save a single pair of example plots, then disable further plotting
            diff = (instance != ecg_arr).astype(int)
            plot_timeseries_with_highlights(
                instance, diff, "perturbation_ecg_highlighted_example.png", variate_labels
            )
            plot_timeseries_differences(
                instance, ecg_arr, "perturbation_ecg_differences_example.png", variate_labels
            )
            flag = False
        samples.append(ecg_arr)

    return np.array(samples)


def perturb_with_slices(
    instance: np.ndarray,
    n_samples: int,
    n_slices: int,
    slices_width: int,
    variate_labels: list[str] | None,
    prob: float = 0.35,
    probs: np.ndarray | None = None,
    flag: bool = True
) -> np.ndarray:
    """Perturb time series by replacing random slices with their local mean.

    Args:
        instance: Base instance of shape (n_variates, n_timesteps).
        n_samples: Number of perturbed samples to generate.
        n_slices: Number of equal-width slices per variate.
        slices_width: Width (in timesteps) of each slice.
        variate_labels: Optional labels for plotting; used if flag=True.
        prob: Probability of perturbing each slice independently.
        flag: If True, saves one example of highlights/differences.

    Returns:
        np.ndarray: Array with shape (n_samples, n_variates, n_timesteps).
    """
    samples = []
    for _ in range(n_samples):
        ecg = []
        for i, lead in enumerate(instance):
            lead_sample = copy.deepcopy(lead)
            for slice_idx in range(n_slices):
                start_idx = slice_idx * slices_width
                end_idx = min(start_idx + slices_width, instance.shape[1])
                if probs is not None:
                    perturb_slice = np.random.binomial(1, abs(probs[i, start_idx:end_idx]).mean())
                else:
                    perturb_slice = np.random.binomial(1, prob)
                if perturb_slice != 1:
                    continue
                if start_idx >= end_idx:
                    continue
                segment = lead_sample[start_idx:end_idx]
                if segment.size == 0:
                    continue
                lead_sample[start_idx:end_idx] = np.mean(segment)
            ecg.append(lead_sample)
        ecg_arr = np.array(ecg)
        if flag:
            diff = (instance != ecg_arr).astype(int)
            plot_timeseries_with_highlights(
                instance, diff, "perturbation_with_slices_highlights_example.png", variate_labels
            )
            plot_timeseries_differences(
                instance,
                ecg_arr,
                "perturbation_with_slices_differences_example.png",
                variate_labels
            )
            flag = False
        samples.append(ecg_arr)
    return np.array(samples)


def perturb_general(
    instance: np.ndarray,
    n_samples: int,
    max_range: float,
    variate_labels: list[str] | None,
    prob: float = 0.35,
    probs: np.ndarray | None = None,
    flag: bool = True
) -> np.ndarray:
    """Apply generic value perturbations with bounded random offsets.

    Each position has an independent probability `prob` of being perturbed by
    adding a random offset sampled uniformly from [x - max_range, x + max_range].

    Args:
        instance: Base instance of shape (n_variates, n_timesteps).
        n_samples: Number of perturbed samples to generate.
        max_range: Half-width of the uniform sampling interval for offsets.
        variate_labels: Optional labels for plotting; used if flag=True.
        prob: Probability of perturbing each position.
        flag: If True, saves one example of highlights/differences.

    Returns:
        np.ndarray: Array with shape (n_samples, n_variates, n_timesteps).
    """
    samples = []
    for _ in range(n_samples):
        if probs is not None:
            perturbation_mask = np.random.binomial(1, abs(probs), size=instance.shape)
        else:
            perturbation_mask = np.random.binomial(1, prob, size=instance.shape)
        # Initialize once and apply all modifications
        sample = copy.deepcopy(instance)
        for i in range(instance.shape[0]):
            for j in range(instance.shape[1]):
                if perturbation_mask[i, j] != 1:
                    continue
                offset = random.uniform(-max_range, max_range)  # noqa: S311
                sample[i, j] += offset
        if flag:
            diff = (instance != sample).astype(int)
            plot_timeseries_with_highlights(
                instance, diff, "perturbation_general_highlights_example.png", variate_labels
            )
            plot_timeseries_differences(
                instance, sample, "perturbation_general_differences_example.png", variate_labels
            )
            flag = False
        samples.append(sample)
    return np.array(samples)
