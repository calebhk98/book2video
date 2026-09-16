from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_text_file(path: str | Path) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8")


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(data, "model_dump"):
        payload = data.model_dump(mode="json")
    else:
        payload = data
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def choose_file_dialog() -> str | None:
    # A tiny GUI picker is attempted here as an optional convenience on desktop Linux.
    # Headless runs can continue to use --chapter-path instead.
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    try:
        selected = filedialog.askopenfilename(
            title="Select chapter",
            filetypes=[("Text / Markdown", "*.txt *.md"), ("All files", "*")],
        )
        return selected or None
    finally:
        root.destroy()
