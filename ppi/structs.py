import dataclasses as dc

import json2dc


@json2dc.add_from_json
@dc.dataclass(kw_only=True)
class PFILocalConfig(json2dc.AddFromJsonInterface):
    max_range: float

    def __post_init__(self) -> None:
        if self.max_range is None:
            raise ValueError("max_range must be provided.")