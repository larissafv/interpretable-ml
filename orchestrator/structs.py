"""Data structures for orchestrator configuration.

This module defines the configuration structures used by the Orchestrator
to manage different explanation methods and their parameters. The configuration
uses dataclasses with JSON serialization support for easy configuration
management.

Classes:
    OrchestratorConfig: Main configuration class for the orchestrator that
        contains method selection, labels, and all sub-configurations.

Typical usage example:

    from orchestrator.structs import OrchestratorConfig
    
    config = OrchestratorConfig(
        method_name="lime",
        variate_labels=["feature1", "feature2"],
        class_labels=["class_a", "class_b"],
        instances_config={"normalize": True},
        model_config={"model_path": "model.h5"},
        method_config={"n_samples": 1000}
    )
    
    # Or load from JSON
    config = OrchestratorConfig.from_json(json_data)
"""

import dataclasses as dc
from typing import Any

import json2dc


@json2dc.add_from_json
@dc.dataclass(kw_only=True)
class OrchestratorConfig(json2dc.AddFromJsonInterface):
    """Configuration class for the Orchestrator.

    This dataclass contains all configuration parameters needed to run
    different explanation methods through the orchestrator. It includes
    method selection, optional labels for interpretability, and nested
    configurations for instances, models, and method-specific parameters.

    The class supports JSON serialization/deserialization for easy
    configuration management and validation of supported methods.

    Attributes:
        method_name: Name of the explanation method to execute.
            Must be one of the supported methods.
        variate_labels: Optional labels for input variables/features.
            Used for plot annotations and result interpretability.
        class_labels: Optional labels for output classes.
            Used for multi-class explanation visualization.
        instances_config: Configuration dictionary for instance formatting.
            Contains parameters for data preprocessing and normalization.
        model_config: Configuration dictionary for model setup.
            Contains model path, parameters, and initialization settings.
        method_config: Configuration dictionary for the selected method.
            Contains method-specific hyperparameters and settings.

    Supported Methods:
        - "lime": Local Interpretable Model-agnostic Explanations
        - "permutation_feature_importance": Global feature importance
        - "global_surrogate": Global interpretable surrogate models
        - "counterfactual_explanations": Counterfactual instance generation

    Raises:
        ValueError: If an unsupported method_name is provided during initialization.

    Note:
        The validation of method_name occurs in __post_init__ to ensure
        only supported methods are used.
    """
    method_name: str
    variate_labels: list[str] | None
    class_labels: list[str] | None
    instances_config: dict[str, Any]
    model_config: dict[str, Any]
    method_config: dict[str, Any]

    def __post_init__(self) -> None:
        """Validates the configuration after initialization.

        Performs validation to ensure that only supported explanation
        methods are specified. This prevents runtime errors and provides
        clear feedback about available options.

        Raises:
            ValueError: If the specified method_name is not in the list
                of supported methods.

        Note:
            This method is automatically called by the dataclass after
            all fields have been initialized.
        """
        # Define supported explanation methods
        supported_methods = [
            "lime",
            "permutation_feature_importance",
            "pfi_local",
            "global_surrogate",
            "counterfactual_explanations",
            "counterfactual_with_pfi",
        ]
        
        # Validate method name
        if self.method_name not in supported_methods:
            raise ValueError(f"Method {self.method_name} is not supported.")
