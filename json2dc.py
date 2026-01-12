"""Minimal local shim for json2dc used in this project.

Provides a no-op decorator and a base class with a from_json constructor
that passes the dict to dataclass constructor. This mirrors the usage in
this repository where dataclasses are decorated with @json2dc.add_from_json
and inherit from AddFromJsonInterface.

If you later add the real json2dc package, remove this file and add the
package to requirements.txt.
"""
from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Type, TypeVar

T = TypeVar("T")


def add_from_json(cls: Type[T]) -> Type[T]:
    """Decorator that injects a simple from_json method for dataclasses."""
    if not is_dataclass(cls):
        # Leave untouched if not a dataclass to avoid surprising behavior
        return cls  # type: ignore[return-value]

    @classmethod  # type: ignore[misc]
    def from_json(cls_: Type[T], data: dict[str, Any]) -> T:  # noqa: N805
        return cls_(**data)

    cls.from_json = from_json  # type: ignore[attr-defined]
    return cls


class AddFromJsonInterface:
    """Marker base class; functionality provided by decorator above."""
    pass
