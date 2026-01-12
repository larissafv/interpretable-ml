"""Module for machine learning model management and prediction.

This module provides a wrapper for TensorFlow/Keras models with configurable
compilation and prediction settings. It handles model loading, compilation
with custom optimizers, and provides a unified interface for predictions
across different explanation methods.

The module supports both pre-compiled models and runtime compilation with
custom loss functions and optimizers. It includes a registry system for
managing different optimizer configurations.

Classes:
    Model: Main wrapper class for TensorFlow/Keras models.

Global Variables:
    _REGISTRY: Registry of available optimizers for model compilation.

Typical usage example:

    from model import Model
    from model.structs import ModelConfig
    
    config = ModelConfig(
        batch_size=32,
        verbose=0,
        compile=False,
        loss="binary_crossentropy",
        optimizer="Adam"
    )
    
    model = Model(config)
    predictions = model.predict(input_data)
"""

from typing import Final

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from common.conf import MODEL_PATH
from model.structs import ModelConfig

# Registry of available optimizers for model compilation
# This allows for easy extension with different optimizer configurations
_REGISTRY: Final[dict[str, object]] = {"Adam": Adam()}


class Model:
    """Wrapper class for TensorFlow/Keras machine learning models.

    This class provides a configurable interface for loading and using
    TensorFlow/Keras models with customizable compilation settings. It supports
    both pre-compiled models and runtime compilation with custom loss functions
    and optimizers.

    The class is designed to work seamlessly with different explanation methods
    by providing a consistent prediction interface while allowing for flexible
    model configuration based on specific requirements.

    Key Features:
        - Configurable batch processing for efficient predictions
        - Runtime compilation with custom loss functions and optimizers
        - Optimizer registry system for easy extension
        - Unified prediction interface for all explanation methods

    Attributes:
        _batch_size: Batch size for prediction operations.
        _verbose: Verbosity level for model operations (0=silent, 1=progress bar, 2=one line).
        _compile: Whether the model should be compiled during initialization.
        _loss: Loss function to use if compilation is needed.
        _optimizer: Optimizer name to use if compilation is needed.
        _model: The loaded TensorFlow/Keras model instance.

    Note:
        If compile=False, both loss and optimizer must be provided for
        runtime compilation. The model path is loaded from common.conf.MODEL_PATH.
    """

    def __init__(self, config: ModelConfig, model_path: str | None = None) -> None:
        """Initializes the Model with configuration settings.

        Loads the model from the configured path and handles compilation
        based on the provided configuration. If compile=False, the model
        will be compiled at runtime with the specified loss and optimizer.

        Args:
            config: Configuration object containing model settings including
                batch size, verbosity, compilation flags, and optimization parameters.

        Raises:
            ValueError: If compile=False but loss or optimizer are not provided
                (this validation occurs in ModelConfig.__post_init__).
            
        Note:
            The model is loaded from the path specified in common.conf.MODEL_PATH.
            Available optimizers are defined in the _REGISTRY dictionary.
        """
        # Store configuration parameters
        self._batch_size = config.batch_size
        self._verbose = config.verbose
        self._compile = config.compile
        self._loss = config.loss
        self._optimizer = config.optimizer

        # Load model from provided path (uploaded) or fallback to configured path
        resolved_path = model_path if model_path else MODEL_PATH
        self._model = load_model(resolved_path, compile=self._compile)

        # Compile model if needed with custom settings
        if not self._compile:
            self._model.compile(
                loss=self._loss,
                optimizer=_REGISTRY.get(self._optimizer, None),
            )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generates predictions for the input data.

        Performs batch prediction on the input data using the configured
        batch size and verbosity settings. This method provides a consistent
        interface for all explanation methods that need model predictions.

        The prediction process respects the configured batch size to optimize
        memory usage and performance, especially important when processing
        large datasets or when used by explanation methods that generate
        many perturbed samples.

        Args:
            x: Input data array for prediction. Shape should match the
                model's expected input format (e.g., (n_samples, n_features)
                for tabular data or (n_samples, height, width, channels)
                for image data).

        Returns:
            np.ndarray: Model predictions with shape (n_samples, n_outputs).
                For binary classification, typically (n_samples, 1).
                For multi-class classification, (n_samples, n_classes).
                For regression, (n_samples, n_targets).

        Note:
            The method uses the batch_size and verbose settings configured
            during initialization. Larger batch sizes may improve performance
            but require more memory.
        """
        y_pred = self._model.predict(
            x, 
            batch_size=self._batch_size, 
            verbose=self._verbose
        )
        return y_pred
