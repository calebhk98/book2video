from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field


class CharacterAsset(BaseModel):
    id: str
    display_name: str
    aliases: list[str] = Field(default_factory=list)
    profile_text: str = ""
    profile_path: str | None = None
    profile_heading: str | None = None
    reference_images: list[str] = Field(default_factory=list)
    voice_reference_files: list[str] = Field(default_factory=list)
    movement_notes: str = ""


class LocationAsset(BaseModel):
    id: str
    display_name: str
    aliases: list[str] = Field(default_factory=list)
    profile_text: str = ""
    profile_path: str | None = None
    profile_heading: str | None = None
    reference_images: list[str] = Field(default_factory=list)


class AssetRegistryData(BaseModel):
    characters: list[CharacterAsset] = Field(default_factory=list)
    locations: list[LocationAsset] = Field(default_factory=list)


class AssetRegistry:
    def __init__(self, data: AssetRegistryData):
        self.data = data
        self.characters = {x.id: x for x in data.characters}
        self.locations = {x.id: x for x in data.locations}

        self._character_aliases: list[tuple[str, str]] = []
        self._location_aliases: list[tuple[str, str]] = []
        for item in data.characters:
            for alias in [item.display_name, item.id, *item.aliases]:
                self._character_aliases.append((alias.casefold(), item.id))
        for item in data.locations:
            for alias in [item.display_name, item.id, *item.aliases]:
                self._location_aliases.append((alias.casefold(), item.id))

        # Longer aliases are checked first as a possible way to avoid matching "Mom"
        # before something more specific such as "Chloe's mom".
        self._character_aliases.sort(key=lambda x: len(x[0]), reverse=True)
        self._location_aliases.sort(key=lambda x: len(x[0]), reverse=True)

    @classmethod
    def empty(cls) -> "AssetRegistry":
        return cls(AssetRegistryData())

    @classmethod
    def from_path(cls, path: str | Path | None) -> "AssetRegistry":
        if not path:
            return cls.empty()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(AssetRegistryData.model_validate(data))

    @staticmethod
    def _markdown_section(text: str, heading: str) -> str:
        # Heading extraction is offered here as a possible way to keep one canonical
        # character bible file while still passing only the relevant character section.
        lines = text.splitlines()
        target = heading.strip().casefold()
        start = None
        level = None
        collected: list[str] = []
        for i, line in enumerate(lines):
            match = re.match(r"^(#{1,6})\s+(.*?)\s*$", line)
            if not match:
                continue
            if match.group(2).strip().casefold() == target:
                start = i + 1
                level = len(match.group(1))
                break
        if start is None or level is None:
            return text
        for line in lines[start:]:
            match = re.match(r"^(#{1,6})\s+", line)
            if match and len(match.group(1)) <= level:
                break
            collected.append(line)
        return "\n".join(collected).strip()

    def _profile_text(self, item) -> str:
        text = item.profile_text
        if item.profile_path and Path(item.profile_path).exists():
            text = Path(item.profile_path).read_text(encoding="utf-8")
            if item.profile_heading:
                text = self._markdown_section(text, item.profile_heading)
        return text

    def resolve_characters(self, text: str) -> list[str]:
        lowered = text.casefold()
        found: list[str] = []
        for alias, item_id in self._character_aliases:
            pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
            if re.search(pattern, lowered) and item_id not in found:
                found.append(item_id)
        return found

    def resolve_location(self, text: str, default: str | None = None) -> str | None:
        lowered = text.casefold()
        for alias, item_id in self._location_aliases:
            pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
            if re.search(pattern, lowered):
                return item_id
        return default

    def character_context(self, ids: list[str]) -> str:
        chunks: list[str] = []
        for item_id in ids:
            item = self.characters.get(item_id)
            if not item:
                continue
            text = self._profile_text(item)
            chunks.append(
                f"CHARACTER {item.id} / {item.display_name}\n{text}\nMovement: {item.movement_notes}".strip()
            )
        return "\n\n".join(chunks)

    def location_context(self, item_id: str | None) -> str:
        if not item_id:
            return ""
        item = self.locations.get(item_id)
        if not item:
            return ""
        text = item.profile_text
        if item.profile_path and Path(item.profile_path).exists():
            text = Path(item.profile_path).read_text(encoding="utf-8")
        return f"LOCATION {item.id} / {item.display_name}\n{text}".strip()

    def character_images(self, ids: list[str]) -> list[str]:
        out: list[str] = []
        for item_id in ids:
            item = self.characters.get(item_id)
            if item:
                out.extend(item.reference_images)
        return list(dict.fromkeys(out))

    def voice_refs(self, ids: list[str]) -> list[str]:
        out: list[str] = []
        for item_id in ids:
            item = self.characters.get(item_id)
            if item:
                out.extend(item.voice_reference_files)
        return list(dict.fromkeys(out))

    def location_images(self, item_id: str | None) -> list[str]:
        if not item_id or item_id not in self.locations:
            return []
        return self.locations[item_id].reference_images
