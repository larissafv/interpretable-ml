"""Module for generating LIME explanations for machine learning models.

This module implements LIME (Local Interpretable Model-agnostic Explanations),
a technique for explaining individual predictions of any machine learning model.
LIME works by learning an interpretable model locally around the prediction
being explained.

The implementation supports different types of data perturbation strategies:
- ECG-specific perturbations for electrocardiogram data
- Slice-based perturbations for time series data
- General perturbations for various data types

Classes:
    LIME: Main class that implements the LIME explanation algorithm.

Typical usage example:

    from lime import LIME
    from lime.structs import LIMEConfig
    
    config = LIMEConfig(lime_type="ecg", n_samples=1000)
    lime_explainer = LIME(model, config)
    lime_explainer.explain(instance, format_instance, variate_labels, class_labels)
"""

from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

from common.perturbations import (
    perturb_ecg,
    perturb_general,
    perturb_with_slices,
    plot_timeseries_with_highlights,
)
from common.views import plot_timeseries_with_heatmap
from instances.format_instance import FormatInstance
from lime.structs import LIMEConfig
from model.model import Model


class LIME:
    """Local Interpretable Model-agnostic Explanations (LIME) implementation.

    LIME is a technique for explaining the predictions of any classifier by learning
    an interpretable model locally around the prediction being explained. It works by:
    
    1. Perturbing the input data around the instance of interest
    2. Getting predictions from the black-box model on perturbed samples
    3. Training a simple interpretable model (linear regression) on these samples
    4. Using the interpretable model's coefficients as feature importance explanations

    The implementation supports multiple perturbation strategies for different data types:
    - ECG: Specialized perturbations for electrocardiogram time series
    - With slices: Time series perturbations using fixed-width segments
    - General: Standard perturbations applicable to various data types

    Attributes:
        _model: The machine learning model to explain.
        _lime_type: Type of perturbation strategy to use.
        _n_slices: Number of slices for slice-based perturbations.
        _sampling_rate: Sampling rate for ECG perturbations.
        _max_range: Maximum range for perturbations.
        _n_samples: Number of perturbed samples to generate.

    Note:
        This implementation uses linear regression as the interpretable model,
        which assumes locally linear behavior around the explained instance.
    """

    def __init__(self, model: Model, config: LIMEConfig) -> None:
        """Initializes the LIME explainer with model and configuration.

        Args:
            model: The machine learning model to generate explanations for.
                Must implement a predict() method that returns predictions.
            config: Configuration object containing LIME hyperparameters
                including perturbation type, number of samples, and other
                strategy-specific parameters.
        """
        # Store model reference
        self._model = model
        
        # Perturbation strategy configuration
        self._lime_type = config.lime_type
        self._n_slices = config.n_slices
        self._sampling_rate = config.sampling_rate
        self._max_range = config.max_range
        
        # Sample generation parameters
        self._n_samples = config.n_samples

        # Number of top features to highlight in plots
        self._top_features_percentage = config.top_features_percentage

    def explain(
        self,
        instance: np.ndarray,
        format_instance: FormatInstance,
        variate_labels: list[str] | None,
        class_labels: list[str] | None,
        counterfactual_with_lime: bool
    ) -> np.ndarray | None:
        """Generates LIME explanations for a given instance.

        This method implements the complete LIME explanation pipeline:
        1. Generates perturbed samples around the target instance
        2. Obtains model predictions for all perturbed samples
        3. Trains a linear regression model on the perturbed data
        4. Generates and saves visualization plots showing feature importance

        The method creates different plots based on whether class labels are provided:
        - Single plot if no class labels (for regression or binary classification)
        - Separate plots for each class if class labels are provided (multi-class)

        Args:
            instance: The input instance to explain. Should be a formatted array
                ready for perturbation (e.g., normalized, preprocessed).
            thresholds: Optional decision thresholds per class or target used by
                downstream consumers. Not used directly in LIME fitting but kept
                for compatibility with calling code.
            format_instance: Formatter object that can convert between formatted
                and model-ready representations of instances.
            variate_labels: Optional labels for variables/features in the data.
                Used for plot annotations and interpretability.
            class_labels: Optional labels for output classes. If provided,
                generates separate explanation plots for each class.

        Returns:
            None: The method generates visualization files as side effects.

        Note:
            Generated plots are saved to the current working directory with
            names like "lime_explanation.png" or "lime_explanation_{class}.png".
        """
        # Generate perturbed samples around the target instance
        samples = self._generate_samples(instance[0], variate_labels)
        
        # Convert samples to model input format and get predictions
        input_model_samples = format_instance.unformat(samples)
        y_pred = self._model.predict(input_model_samples)
        
        # Keep first dimension (n_samples) and flatten the last two (features)
        n_samples, n_variates, n_timesteps = samples.shape
        x_flat = samples.reshape(n_samples, n_variates * n_timesteps)
        
        # Train interpretable linear model on flattened perturbed data
        interpretable_model = LinearRegression()
        interpretable_model.fit(X=x_flat, y=y_pred)
        
        # Reshape coefficients back to (targets, variates, timesteps) or (variates, timesteps)
        coef = interpretable_model.coef_
        top_points = self._get_top_features(coef)

        if coef.ndim == 1:
            # Single target/regression: shape -> (n_variates, n_timesteps)
            coef_reshaped = coef.reshape(n_variates, n_timesteps)
            top_points = top_points.reshape(n_variates, n_timesteps)
        else:
            # Multi-target: shape -> (n_targets, n_variates, n_timesteps)
            expected_features = n_variates * n_timesteps
            if coef.shape[1] != expected_features:
                msg = (
                    "Unexpected coef shape {}; expected (*, {}) given samples shape {}"
                ).format(
                    coef.shape, expected_features, (n_samples, n_variates, n_timesteps)
                )
                raise ValueError(msg)
            coef_reshaped = coef.reshape(coef.shape[0], n_variates, n_timesteps)
            top_points = top_points.reshape(top_points.shape[0], n_variates, n_timesteps)

        # Generate visualizations based on class configuration
        if class_labels is None:
            # Single plot for binary classification or regression
            plot_name_heatmap = Path(
                f"lime_explanation_{self._lime_type}_{self._n_samples}_heatmap.png"
            )
            plot_timeseries_with_heatmap(
                instance[0], coef_reshaped, plot_name_heatmap, variate_labels
            )
            
            plot_name_highlight = Path(
                f"lime_explanation_{self._lime_type}_{self._n_samples}_highlights.png"
            )
            plot_timeseries_with_highlights(
                instance[0], top_points, plot_name_highlight, variate_labels
            )
            if counterfactual_with_lime:
                norm_coef = self._normalize_array(coef_reshaped)
                return norm_coef
            return None
        
        # Multiple plots for multi-class classification
        for i, class_label in enumerate(class_labels):
            plot_name_heatmap = Path(
                f"lime_explanation_{self._lime_type}_{self._n_samples}_{class_label}_heatmap.png"
            )
            plot_timeseries_with_heatmap(
                instance[0], coef_reshaped[i], plot_name_heatmap, variate_labels, title=class_label
            )

            plot_name_highlight = Path(
                f"lime_explanation_{self._lime_type}_{self._n_samples}_{class_label}_highlights.png"
            )
            plot_timeseries_with_highlights(
                instance[0], top_points[i], plot_name_highlight, variate_labels, title=class_label
            )
        if counterfactual_with_lime:
            norm_coef = self._normalize_array(coef_reshaped)
            return norm_coef
        return None

    def _generate_samples(
        self, instance: np.ndarray, variate_labels: list[str] | None
    ) -> np.ndarray:
        """Generates perturbed samples around the target instance.

        Creates a set of perturbed versions of the input instance using different
        perturbation strategies based on the configured LIME type. The perturbations
        are designed to preserve the local neighborhood while introducing controlled
        variations for learning feature importance.

        Args:
            instance: The original instance to perturb. Shape and format depend
                on the perturbation strategy being used.
            variate_labels: Optional labels for variables/features, used by some
                perturbation strategies for domain-specific logic.

        Returns:
            np.ndarray: Array of perturbed samples with shape (n_samples, *instance.shape).
                Each row represents a perturbed version of the original instance.

        Raises:
            ValueError: If an unsupported lime_type is specified in the configuration.

        Note:
            Different perturbation strategies are suitable for different data types:
            - "ecg": Optimized for electrocardiogram time series data
            - "with_slices": Segments time series into fixed-width slices
            - "general": Generic perturbations for various data types
        """
        if self._lime_type == "ecg":
            # ECG-specific perturbations for electrocardiogram data
            return perturb_ecg(
                instance,
                None,
                self._n_samples,
                self._sampling_rate,
                self._max_range,
                variate_labels
            )
        elif self._lime_type == "with_slices":
            # Slice-based perturbations for time series data
            slices_width = instance.shape[1] // self._n_slices
            return perturb_with_slices(
                instance, self._n_samples, self._n_slices, slices_width, variate_labels
            )
        elif self._lime_type == "general":
            # General perturbations for various data types
            return perturb_general(instance, self._n_samples, self._max_range, variate_labels)

    def _get_top_features(self, coef: np.ndarray) -> np.ndarray:
        n_best_features = int((self._top_features_percentage / 100) * coef.shape[-1])
        mask = []
        if coef.ndim > 1:
            for class_ in coef:
                top_features_idx = []
                for idx in range(class_.shape[0]):
                    if len(top_features_idx) < n_best_features:
                        top_features_idx.append(idx)
                    elif abs(class_[idx]) > abs(class_[top_features_idx[-1]]):
                        top_features_idx[-1] = idx
                    
                    top_features_idx.sort(key=lambda x: abs(class_[x]), reverse=True)

                # Create binary mask
                class_mask = np.zeros_like(class_, dtype=int)
                class_mask[top_features_idx] = 1
                mask.append(class_mask)
        else:
            top_features_idx = []
            for idx in range(coef.shape[0]):
                if len(top_features_idx) <= n_best_features:
                    top_features_idx.append(idx)
                elif abs(coef[idx]) > abs(coef[top_features_idx[-1]]):
                    top_features_idx[-1] = idx
                
                top_features_idx.sort(key=lambda x: abs(coef[x]), reverse=True)

            # Create binary mask
            class_mask = np.zeros_like(coef, dtype=int)
            class_mask[top_features_idx] = 1

        return np.array(mask)

    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Normalizes a NumPy array to the range [0, 1] using Min-Max Normalization.
        """
        if arr.ndim > 1:
            norm_arr = []
            for i in range(arr.shape[0]):
                arr_min = np.min(arr[i])
                arr_max = np.max(arr[i])

                if arr_max == arr_min:  # Handle cases where all values are the same
                    norm_arr.append(np.zeros_like(arr[i], dtype=float))

                norm_arr.append((arr[i] - arr_min) / (arr_max - arr_min))
        else:
            arr_min = np.min(arr)
            arr_max = np.max(arr)

            if arr_max == arr_min:  # Handle cases where all values are the same
                return np.zeros_like(arr, dtype=float)

            norm_arr = (arr - arr_min) / (arr_max - arr_min)
        return np.array(norm_arr)