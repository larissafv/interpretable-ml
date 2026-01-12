"""Configuration structures for Permutation Feature Importance (PFI)."""

import dataclasses as dc

import json2dc


@json2dc.add_from_json
@dc.dataclass(kw_only=True)
class PFIConfig(json2dc.AddFromJsonInterface):
    """PFI configuration dataclass.

    Attributes:
        simplify_algorithm: If True, compute PFI over fixed-width slices.
        n_slices_ts: Number of slices when simplify_algorithm is True.
        feat_max: Upper replacement value for permutation (optional).
        feat_min: Lower replacement value for permutation (optional).
    """
    simplify_algorithm: bool
    n_slices_ts: int | None
    feat_max: float | None
    feat_min: float | None

    def __post_init__(self) -> None:
        if self.simplify_algorithm and self.n_slices_ts is None:
            raise ValueError("n_slices_ts must be provided.")
        if self.feat_max is not None and self.feat_min is None:
            raise ValueError("feat_min must be provided if feat_max is provided.")
        if self.feat_min is not None and self.feat_max is None:
            raise ValueError("feat_max must be provided if feat_min is provided.")