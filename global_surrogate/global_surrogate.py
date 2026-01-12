"""Global surrogate model implementation.

Fits a simple interpretable model (LinearRegression) to approximate the
black-box model's predictions and visualizes learned coefficients as a
heatmap over time for multivariate time series.
"""

from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

from common.views import plot_timeseries_with_heatmap
from instances.format_instance import FormatInstance
from model.model import Model


class GlobalSurrogate:
    """Create a global surrogate model for a black-box model."""

    def __init__(self, model: Model) -> None:
        """Initialize with a black-box model wrapper."""
        self._model = model

    def explain(
        self,
        instances: np.ndarray,
        instance_idx: int,
        format_instance: FormatInstance,
        variate_labels: list[str],
        class_labels: list[str]
    ) -> None:
        """Fit a linear surrogate and plot coefficient heatmaps.

        Args:
            instances: Input batch as formatted array.
            variate_labels: Optional labels for features/variates.
            class_labels: Optional list of output class labels.
        """
        input_model = format_instance.unformat(instances)
        y_pred = self._model.predict(input_model)

        # Keep first dimension (n_instances) and flatten the last two (features)
        n_instances, n_variates, n_timesteps = instances.shape
        x_flat = instances.reshape(n_instances, n_variates * n_timesteps)

        interpretable_model = LinearRegression()
        interpretable_model.fit(X=x_flat, y=y_pred)
        interpretable_model.predict(x_flat)

        # Reshape coefficients back to (targets, variates, timesteps) or (variates, timesteps)
        coef = interpretable_model.coef_
        if coef.ndim == 1:
            # Single target/regression: shape -> (n_variates, n_timesteps)
            coef_reshaped = coef.reshape(n_variates, n_timesteps)
        else:
            # Multi-target: shape -> (n_targets, n_variates, n_timesteps)
            expected_features = n_variates * n_timesteps
            if coef.shape[1] != expected_features:
                msg = (
                    "Unexpected coef shape {}; expected (*, {}) given samples shape {}"
                ).format(
                    coef.shape, expected_features, (n_instances, n_variates, n_timesteps)
                )
                raise ValueError(msg)
            coef_reshaped = coef.reshape(coef.shape[0], n_variates, n_timesteps)

        # Generate visualizations based on class configuration
        if class_labels is None:
            plot_name = Path("global_surrogate.png")
            plot_timeseries_with_heatmap(
                instances[instance_idx], coef_reshaped, plot_name, variate_labels
            )
            return None
        for i, class_label in enumerate(class_labels):
            plot_name = Path(f"global_surrogate_{class_label}.png")
            plot_timeseries_with_heatmap(
                instances[instance_idx],
                coef_reshaped[i],
                plot_name,
                variate_labels,
                title=class_label
            )
        return None
