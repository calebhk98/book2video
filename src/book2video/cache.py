from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class StageCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, stage: str, payload: Any, version: str = "1") -> str:
        # Content-addressed keys are being used here as one possible way to let unchanged
        # scenes reuse downstream work after a chapter edit.
        if isinstance(payload, BaseModel):
            payload = payload.model_dump(mode="json")
        encoded = json.dumps(
            {"stage": stage, "version": version, "payload": payload},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def path_for(self, stage: str, key: str) -> Path:
        return self.root / stage / f"{key}.json"

    def get(self, stage: str, key: str, model: type[BaseModel]) -> BaseModel | None:
        path = self.path_for(stage, key)
        if not path.exists():
            return None
        return model.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, stage: str, key: str, value: BaseModel) -> Path:
        path = self.path_for(stage, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value.model_dump_json(indent=2), encoding="utf-8")
        return path
