"""Data structures for model configuration.

This module defines the configuration structures used by the Model class
to manage TensorFlow/Keras model settings. The configuration supports both
pre-compiled models and runtime compilation scenarios with validation.

Classes:
    ModelConfig: Configuration class for model initialization and compilation.

Typical usage example:

    from model.structs import ModelConfig
    
    # For pre-compiled model
    config = ModelConfig(
        batch_size=32,
        verbose=0,
        compile=True,
        loss=None,
        optimizer=None
    )
    
    # For runtime compilation
    config = ModelConfig(
        batch_size=64,
        verbose=1,
        compile=False,
        loss="binary_crossentropy",
        optimizer="Adam"
    )
    
    # Or load from JSON
    config = ModelConfig.from_json(json_data)
"""

import dataclasses as dc

import json2dc


@json2dc.add_from_json
@dc.dataclass(kw_only=True)
class ModelConfig(json2dc.AddFromJsonInterface):
    """Configuration class for Model initialization and compilation.

    This dataclass contains all parameters needed to configure a TensorFlow/Keras
    model wrapper, including prediction settings and optional compilation parameters.
    It supports both pre-compiled models and runtime compilation scenarios.

    The class includes validation to ensure that compilation parameters are
    provided when needed, preventing runtime errors and ensuring proper
    model configuration.

    Attributes:
        batch_size: Batch size for prediction operations. Larger values may
            improve performance but require more memory. Typical values: 16-128.
        verbose: Verbosity level for model operations.
            0 = silent, 1 = progress bar, 2 = one line per epoch.
        compile: Whether the model should be loaded as pre-compiled.
            True = use existing compilation, False = compile at runtime.
        loss: Loss function name for runtime compilation.
            Required if compile=False. Examples: "binary_crossentropy", "mse".
        optimizer: Optimizer name for runtime compilation.
            Required if compile=False. Must exist in the model registry.

    Validation Rules:
        - If compile=False, both loss and optimizer must be provided
        - If compile=True, loss and optimizer can be None

    Raises:
        ValueError: If compile=False but loss or optimizer are None.

    Note:
        The validation occurs in __post_init__ to ensure configuration
        consistency before the Model class attempts to use the settings.
    """
    batch_size: int
    verbose: int
    compile: bool
    loss: str | None
    optimizer: str | None

    def __post_init__(self) -> None:
        """Validates the configuration after initialization.

        Ensures that runtime compilation parameters are provided when needed.
        This prevents runtime errors when the Model class attempts to compile
        the model with missing parameters.

        Raises:
            ValueError: If compile=False but either loss or optimizer is None.
                Both parameters are required for runtime compilation.

        Note:
            This method is automatically called by the dataclass after
            all fields have been initialized.
        """
        if not self.compile and (self.loss is None or self.optimizer is None):
            raise ValueError("If compile is False, both loss and optimizer must be provided.")
