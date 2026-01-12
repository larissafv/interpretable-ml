"""Configuration structures for instance formatting.

Defines dataclasses with JSON (de)serialization for specifying the number
of variates and timesteps used to format/unformat instances consistently.
"""

import dataclasses as dc

import json2dc


@json2dc.add_from_json
@dc.dataclass(kw_only=True)
class InstanceConfig(json2dc.AddFromJsonInterface):
    """Instance shape configuration for format/unformat utilities."""
    variates: int | None
    timesteps: int

    def __post_init__(self) -> None:
        if self.variates is not None and self.variates < 0:
            raise ValueError("Variates must be a non-negative integer or None.")
        if self.timesteps < 0:
            raise ValueError("Timesteps must be a non-negative integer.")