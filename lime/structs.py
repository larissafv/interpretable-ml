"""Configuration structures for LIME explainability."""

import dataclasses as dc

import json2dc


@json2dc.add_from_json
@dc.dataclass(kw_only=True)
class LIMEConfig(json2dc.AddFromJsonInterface):
    """LIME configuration dataclass.

    Attributes:
        lime_type: One of {"ecg", "with_slices", "general"}.
        n_slices: Number of slices for the "with_slices" strategy.
        sampling_rate: Sampling rate for ECG perturbations.
        max_range: Max perturbation range for "general" strategy.
        n_samples: Number of perturbed samples to generate.
    """
    lime_type: str
    n_slices: int | None
    sampling_rate: int | None
    max_range: float | None
    n_samples: int
    top_features_percentage: int  # Percentage of top features to highlight in plots.

    def __post_init__(self) -> None:
        if self.lime_type not in ["ecg", "with_slices", "general"]:
            raise ValueError(f"LIME type {self.lime_type} is not supported.")
        if self.n_slices is None and self.lime_type == "with_slices":
            raise ValueError("n_slices must be specified for with_slices LIME type.")
        if self.sampling_rate is None and self.lime_type == "ecg":
            raise ValueError("sampling_rate must be specified for ECG LIME type.")
        if self.max_range is None and self.lime_type == "general":
            raise ValueError("max_range must be specified for general LIME type.")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")
