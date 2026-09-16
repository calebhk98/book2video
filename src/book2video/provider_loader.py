from __future__ import annotations

import importlib
from typing import Any


def load_object(path: str) -> type[Any]:
    """Load 'package.module:ClassName' without hard-coding provider imports."""
    module_name, sep, object_name = path.partition(":")
    if not sep:
        raise ValueError(f"Expected 'module:object', got {path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def build_provider(class_path: str, options: dict[str, Any]) -> Any:
    cls = load_object(class_path)
    return cls(**options)
