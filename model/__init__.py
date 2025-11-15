"""Model package for machine learning model management.

This package provides a wrapper for TensorFlow/Keras models with configurable
compilation and prediction settings. It's designed to work seamlessly with
different explanation methods by providing a consistent prediction interface.

Key Features:
    - Configurable model loading and compilation
    - Batch prediction optimization
    - Runtime compilation with custom optimizers
    - Unified interface for all explanation methods

Modules:
    model: Main Model class implementation
    structs: Configuration data structures

The package supports both pre-compiled models and runtime compilation scenarios,
making it flexible for different deployment and explanation requirements.

Example:
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

# Model module
