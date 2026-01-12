"""Instance formatting helpers for model input/output shapes.

`FormatInstance` standardizes shapes between explanation algorithms and the
downstream model, providing reversible format/unformat conversions.
"""

import numpy as np

from .structs import InstanceConfig


class FormatInstance:
    """Utility to convert instances between algorithm and model formats."""
    def __init__(self, config: InstanceConfig) -> None:
        """Store shape-related configuration.

        Args:
            config: Dataclass with `variates` and `timesteps` info.
        """
        self._variates = config.variates
        self._timesteps = config.timesteps

    def format(self, instance: np.ndarray) -> np.ndarray:
        """Format instance(s) to the algorithm-friendly shape.

        Raises descriptive errors on unexpected shapes to avoid silent bugs.
        """
        if instance.ndim > 3:
            raise ValueError(
                f"Instance has too many dimensions: {instance.ndim}. Expected 1D, 2D or 3D."
            )
        if self._variates is None:
            if instance.ndim == 1:
                return instance.reshape(1, instance.shape[0])
            elif instance.ndim == 2:
                return instance
            else:
                raise ValueError("Variates must be specified in the configuration.")
        elif self._timesteps < self._variates:
            if instance.ndim == 2:
                instance = np.transpose(instance)
                return instance.reshape(1, instance.shape[0], instance.shape[1])
            if instance.ndim == 3:
                return np.transpose(np.array(instance), (0, 2, 1))
        return instance

    def unformat(self, instance: np.ndarray) -> np.ndarray:
        """Convert formatted instance(s) back to the model-ready shape.

        Performs the inverse of `format` when `variates` and `timesteps`
        indicate variates-first representations.
        """
        if instance.ndim not in [2, 3]:
            raise ValueError(
                f"Instance has invalid dimensions: {instance.ndim}. Expected 2D or 3D."
            )
        if self._variates is None:
            if instance.ndim == 2:
                return instance
            else:
                raise ValueError("Variates must be specified in the configuration.")
        if self._timesteps < self._variates:
            if instance.ndim != 3:
                raise ValueError(
                    f"Instance has variates but doesn't have 3 dimensions: {instance.ndim}."
                    )
            else:
                return np.transpose(np.array(instance), (0, 2, 1))
        return instance