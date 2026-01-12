"""Module for orchestrating different explainable AI methods.

This module provides a unified interface for running various interpretability
and explainability methods on machine learning models. It acts as a central
coordinator that can execute different explanation techniques based on
configuration parameters.

Supported explanation methods:
- LIME: Local Interpretable Model-agnostic Explanations
- Permutation Feature Importance: Global feature importance through permutation
- Global Surrogate: Interpretable surrogate models for global explanations
- Counterfactual Explanations: Instance-specific counterfactual generation

Classes:
    Orchestrator: Main class that coordinates the execution of explanation methods.

Typical usage example:

    from orchestrator import Orchestrator
    from orchestrator.structs import OrchestratorConfig
    
    config = OrchestratorConfig(
        method_name="lime",
        variate_labels=["feature1", "feature2"],
        class_labels=["class_a", "class_b"],
        instances_config={...},
        model_config={...},
        method_config={...}
    )
    
    orchestrator = Orchestrator(config)
    orchestrator.explain(instances, instance_idx, ground_truth)
"""

import numpy as np

from counterfactual_explanations.counterfactual_explanations import CounterfactualExplanation
from counterfactual_explanations.structs import CEConfig
from global_surrogate.global_surrogate import GlobalSurrogate
from instances.format_instance import FormatInstance
from instances.structs import InstanceConfig
from lime.lime import LIME
from lime.structs import LIMEConfig
from model.model import Model
from model.structs import ModelConfig
from orchestrator.structs import OrchestratorConfig
from permutation_feature_importance.permutation_feature_importance import (
    PermutationFeatureImportance,
)
from permutation_feature_importance.structs import PFIConfig
from ppi.pfi_local import PFILocal
from ppi.structs import PFILocalConfig


class Orchestrator:
    """Central coordinator for executing different explainable AI methods.

    The Orchestrator class provides a unified interface for running various
    explanation and interpretability methods on machine learning models. It handles
    the initialization of different explanation techniques and manages their
    execution with appropriate validation and error handling.

    This design pattern allows for easy switching between explanation methods
    and centralizes the configuration management for complex explanation pipelines.

    Supported Methods:
        - lime: Local explanations using LIME algorithm
        - permutation_feature_importance: Global feature importance via permutation
        - global_surrogate: Global interpretable surrogate models
        - counterfactual_explanations: Instance-specific counterfactual generation

    Attributes:
        _method_name: Name of the explanation method to execute.
        _variate_labels: Optional labels for input variables/features.
        _class_labels: Optional labels for output classes.
        _format_instance: Instance formatter for data preprocessing.
        _model: Machine learning model to explain.
        _method_config: Method-specific configuration parameters.

    Note:
        Different methods have different input requirements:
        - Some require ground truth labels
        - Some require specific instance indices
        - Configuration validation is performed during initialization
    """

    def __init__(self, config: OrchestratorConfig, model_path: str | None = None) -> None:
        """Initializes the Orchestrator with configuration settings.

        Sets up the orchestrator with all necessary components including
        the explanation method, data formatters, model, and method-specific
        configurations.

        Args:
            config: Configuration object containing all settings for the
                orchestrator including method selection, labels, instance
                formatting, model configuration, and method-specific parameters.

        Note:
            The configuration is validated during initialization to ensure
            all required components are properly configured.
        """
        # Store method selection and labels
        self._method_name = config.method_name
        self._variate_labels = config.variate_labels
        self._class_labels = config.class_labels
        
        # Initialize core components
        self._format_instance = FormatInstance(InstanceConfig.from_json(config.instances_config))
        # Allow overriding the model path with an uploaded file if provided
        self._model = Model(
            ModelConfig.from_json(config.model_config),
            model_path=model_path,
        )
        
        # Store method-specific configuration
        self._method_config = config.method_config or {}

        # Split method_config into counterfactual and lime specific configs.
        # Support both nested shapes (method_config.counterfactual / method_config.lime)
        # and flat shapes (top-level keys). Top-level values are used as defaults
        # for both sub-configs and nested dicts override them.
        ce_fields = set(CEConfig.__dataclass_fields__.keys())
        pfi_fields = set(PFILocalConfig.__dataclass_fields__.keys())

        # Start from top-level keys (flat) as defaults
        ce_json = {k: v for k, v in self._method_config.items() if k in ce_fields}
        pfi_json = {k: v for k, v in self._method_config.items() if k in pfi_fields}

        # If nested configs provided explicitly, merge/override
        nested_cf = self._method_config.get('counterfactual')
        if isinstance(nested_cf, dict):
            ce_json.update(nested_cf)

        nested_pfi = self._method_config.get('pfi_local')
        if isinstance(nested_pfi, dict):
            pfi_json.update(nested_pfi)

        # Store prepared JSON dicts for later use in explain()
        self._ce_method_config = ce_json
        self._pfi_local_method_config = pfi_json

    def explain(
            self, instances: np.ndarray, instance_idx: int, ground_truth: np.ndarray = None
        ) -> float | None:
        """Executes the configured explanation method on the provided data.

        This method serves as the main entry point for generating explanations.
        It validates input parameters based on the selected method's requirements,
        then delegates to the appropriate explanation implementation.

        Different methods have different requirements:
        - LIME: Requires valid instance_idx for single instance explanation
        - PFI: Works with all instances, no specific index needed
        - Global Surrogate: Requires ground_truth for model training
        - Counterfactual: Requires both valid instance_idx and ground_truth

        Args:
            instances: Input data array containing instances to explain.
                Shape depends on the data type and model requirements.
            instance_idx: Index of the specific instance to explain.
                Required for LIME and counterfactual explanations.
                Should be None or ignored for global methods.
            ground_truth: True labels corresponding to instances.
                Required for global_surrogate and counterfactual_explanations.
                Optional for other methods.

        Returns:
            None: All methods generate visualization files as side effects.

        Raises:
            ValueError: If required parameters are missing or invalid for the
                selected method, or if an unsupported method is specified.

        Note:
            Generated explanation files are saved to the current working
            directory with method-specific naming conventions.
        """
        # Validate ground truth requirement for specific methods
        if self._method_name in ["permutation_feature_importance",
                                "counterfactual_explanations",
                                "counterfactual_with_pfi"]:
            if ground_truth is None:
                raise ValueError(
                    f"Ground truth should be provided for method {self._method_name}."
                )
        
        # Validate instance index requirement for specific methods
        if self._method_name in ["lime", "counterfactual_explanations", "counterfactual_with_pfi"]:
            if instance_idx is None or not (0 <= instance_idx < len(instances)):
                raise ValueError(
                    f"Valid instance index should be provided for method {self._method_name}."
                )
    
        # Execute the appropriate explanation method
        if self._method_name == "lime":
            # LIME: Local explanations for a single instance
            input_explanation = self._format_instance.format(instances[instance_idx])
            lime = LIME(self._model, LIMEConfig.from_json(self._method_config))
            lime.explain(
                input_explanation,
                self._format_instance,
                self._variate_labels,
                self._class_labels,
                counterfactual_with_lime=False
            )

        elif self._method_name == "permutation_feature_importance":
            # PFI: Global feature importance through permutation testing
            input_explanation = self._format_instance.format(instances)
            pfi = PermutationFeatureImportance(
                self._model, PFIConfig.from_json(self._method_config)
            )
            pfi.explain(
                input_explanation,
                ground_truth,
                self._format_instance,
                self._variate_labels,
                self._class_labels
            )

        elif self._method_name == "pfi_local":
            # PFILocal: local permutation-like importance per timepoint/variate
            # This operates on a single instance (use instance_idx)
            input_explanation = self._format_instance.format(instances[instance_idx])
            pfil = PFILocal(self._model, PFILocalConfig.from_json(self._method_config))
            pfil.explain(
                input_explanation,
                self._format_instance,
                self._variate_labels,
                self._class_labels,
                counterfactual_with_pfi=False
            )

        elif self._method_name == "global_surrogate":
            # Global Surrogate: Interpretable model approximating the original
            input_explanation = self._format_instance.format(instances)
            surrogate = GlobalSurrogate(self._model)
            surrogate.explain(
                input_explanation,
                instance_idx,
                self._format_instance,
                self._variate_labels,
                self._class_labels
            )

        elif self._method_name == "counterfactual_explanations":
            # Counterfactual: Generate alternative instances with different predictions
            ce = CounterfactualExplanation(self._model, CEConfig.from_json(self._ce_method_config))
            fitness = ce.explain(
                instances,
                instance_idx,
                ground_truth,
                self._format_instance,
                self._variate_labels,
                self._class_labels
            )
            return fitness
        elif self._method_name == "counterfactual_with_pfi":
            input_explanation = self._format_instance.format(instances[instance_idx])

            # Build PFILocal and CE objects using their separated configs
            pfil = PFILocal(self._model, PFILocalConfig.from_json(self._pfi_local_method_config))
            probs = pfil.explain(
                input_explanation,
                self._format_instance,
                self._variate_labels,
                self._class_labels,
                counterfactual_with_pfi=True
            )

            ce = CounterfactualExplanation(self._model, CEConfig.from_json(self._ce_method_config))
            fitness = ce.explain(
                instances,
                instance_idx,
                ground_truth,
                self._format_instance,
                self._variate_labels,
                self._class_labels,
                probs=probs
            )
            return fitness

        return None