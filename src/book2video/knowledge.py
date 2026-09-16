from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    source: str
    text: str
    terms: set[str]


class LocalKnowledgeIndex:
    """A deliberately small local retriever for profile/world documents."""

    def __init__(self, paths: list[str]):
        self.chunks: list[Chunk] = []
        for raw in paths:
            path = Path(raw)
            if path.is_dir():
                for child in path.rglob("*"):
                    if child.suffix.lower() in {".md", ".txt"}:
                        self._add_file(child)
            elif path.exists() and path.suffix.lower() in {".md", ".txt"}:
                self._add_file(path)

    @staticmethod
    def _terms(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9']{3,}", text.casefold()))

    def _add_file(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        # Paragraph-sized chunks are being tried here as a lightweight alternative to
        # bringing in a vector database before the project proves it needs one.
        for part in re.split(r"\n\s*\n", text):
            part = part.strip()
            if len(part) < 20:
                continue
            self.chunks.append(Chunk(str(path), part, self._terms(part)))

    def search(self, query: str, top_k: int = 6) -> list[str]:
        q = self._terms(query)
        if not q:
            return []
        scored: list[tuple[float, Chunk]] = []
        for chunk in self.chunks:
            overlap = len(q & chunk.terms)
            if overlap:
                score = overlap / max(1, len(q))
                scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f"[{c.source}]\n{c.text}" for _, c in scored[:top_k]]
