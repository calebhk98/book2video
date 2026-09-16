from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from book2video.media import ffmpeg_available
from book2video.providers.base import StructuredLLMRequest, VideoGenerationRequest


def env_value(name: str, explicit: str | None = None) -> str:
    """Resolve a secret while keeping provider configuration free to override environment lookup."""
    if explicit:
        return explicit
    value = os.getenv(name, "")
    if not value:
        raise RuntimeError(f"Missing credential: set {name} or pass it in provider options.")
    return value


def build_structured_prompt(request: StructuredLLMRequest, response_model: type[BaseModel]) -> str:
    # Supplying both context and schema in plain text is being tried as a compatibility layer.
    # Native schema controls still remain the authoritative constraint when a provider supports them.
    schema = response_model.model_json_schema()
    return (
        f"Stage: {request.stage}\n\n"
        f"Instructions:\n{request.instructions.strip()}\n\n"
        "Context JSON:\n"
        f"{json.dumps(request.context, ensure_ascii=False, indent=2)}\n\n"
        "Return only data matching this JSON Schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


def parse_json_text(text: str) -> Any:
    """Accept plain JSON and a small amount of common markdown wrapping from less strict models."""
    cleaned = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.I | re.S)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Extracting the widest apparent object/array is attempted only as a fallback; validation still follows.
        starts = [(cleaned.find("{"), "{"), (cleaned.find("["), "[")]
        starts = [(i, ch) for i, ch in starts if i >= 0]
        if not starts:
            raise
        start, ch = min(starts, key=lambda x: x[0])
        end_ch = "}" if ch == "{" else "]"
        end = cleaned.rfind(end_ch)
        if end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def mime_type(path: str | Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def file_to_base64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def file_to_data_uri(path: str | Path) -> str:
    return f"data:{mime_type(path)};base64,{file_to_base64(path)}"


def _video_duration(path: str | Path) -> float | None:
    if not ffmpeg_available():
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def sample_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    count: int = 6,
    max_width: int = 960,
) -> list[str]:
    """Create a few approximately-even frame samples for LLM backends without native video input."""
    if count <= 0 or not ffmpeg_available():
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = _video_duration(video_path) or float(count)
    fps = max(0.01, count / max(duration, 0.1))
    pattern = output_dir / "frame_%03d.jpg"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps:.8f},scale='min({max_width},iw)':-2",
        "-frames:v",
        str(count),
        "-q:v",
        "3",
        str(pattern),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if proc.returncode != 0:
        return []
    return [str(p) for p in sorted(output_dir.glob("frame_*.jpg"))[:count]]


def materialize_attachment_images(
    request: StructuredLLMRequest,
    temp_dir: str | Path,
    *,
    video_frame_samples: int = 6,
    max_width: int = 960,
) -> list[tuple[str, str]]:
    """Return (label, image_path) pairs; videos are represented by sampled frames."""
    temp_dir = Path(temp_dir)
    images: list[tuple[str, str]] = []
    for index, attachment in enumerate(request.attachments):
        source = Path(attachment.path)
        label = attachment.label or f"attachment {index + 1}"
        if attachment.kind == "image" or mime_type(source).startswith("image/"):
            if source.exists():
                images.append((label, str(source)))
            continue
        if attachment.kind == "video" or mime_type(source).startswith("video/"):
            sampled = sample_video_frames(
                source,
                temp_dir / f"video_{index:02d}",
                count=video_frame_samples,
                max_width=max_width,
            )
            images.extend((f"{label} frame {i + 1}/{len(sampled)}", p) for i, p in enumerate(sampled))
    return images


def recursive_find_first_url(value: Any) -> str | None:
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        return value
    if isinstance(value, list):
        for item in value:
            found = recursive_find_first_url(item)
            if found:
                return found
    if isinstance(value, dict):
        # Looking at likely media keys first may avoid choosing a status URL from provider metadata.
        for key in ("video", "videos", "output", "url", "uri", "file"):
            if key in value:
                found = recursive_find_first_url(value[key])
                if found:
                    return found
        for item in value.values():
            found = recursive_find_first_url(item)
            if found:
                return found
    return None


def get_by_path(value: Any, path: str | None) -> Any:
    if not path:
        return value
    current = value
    for piece in path.split("."):
        if isinstance(current, list):
            current = current[int(piece)]
        elif isinstance(current, dict):
            current = current[piece]
        else:
            raise KeyError(path)
    return current


def render_template(value: Any, variables: dict[str, Any]) -> Any:
    """Recursively replace {{NAME}} placeholders while preserving list/dict types for whole-value placeholders."""
    if isinstance(value, list):
        return [render_template(x, variables) for x in value]
    if isinstance(value, dict):
        return {k: render_template(v, variables) for k, v in value.items()}
    if not isinstance(value, str):
        return value

    exact = re.fullmatch(r"\{\{([A-Z0-9_]+)\}\}", value)
    if exact:
        return variables.get(exact.group(1), value)

    def replace(match: re.Match[str]) -> str:
        replacement = variables.get(match.group(1), match.group(0))
        if isinstance(replacement, (dict, list)):
            return json.dumps(replacement, ensure_ascii=False)
        return str(replacement)

    return re.sub(r"\{\{([A-Z0-9_]+)\}\}", replace, value)


def video_template_variables(
    request: VideoGenerationRequest,
    devices: tuple[str, ...],
    *,
    include_data_uris: bool = True,
) -> dict[str, Any]:
    # Data-URI creation can be surprisingly expensive with large character sheets, so local wrappers can skip it.
    if include_data_uris:
        char_uris = [file_to_data_uri(p) for p in request.character_reference_images if Path(p).exists()]
        location_uris = [file_to_data_uri(p) for p in request.location_reference_images if Path(p).exists()]
        voice_uris = [file_to_data_uri(p) for p in request.voice_reference_files if Path(p).exists()]
        continuation_video = (
            file_to_data_uri(request.continuation_video_path)
            if request.continuation_video_path and Path(request.continuation_video_path).exists()
            else None
        )
        continuation_frame = (
            file_to_data_uri(request.continuation_frame_path)
            if request.continuation_frame_path and Path(request.continuation_frame_path).exists()
            else None
        )
    else:
        char_uris = []
        location_uris = []
        voice_uris = []
        continuation_video = None
        continuation_frame = None
    all_refs = char_uris + location_uris
    return {
        "CLIP_ID": request.clip_id,
        "TAKE_NUMBER": request.take_number,
        "DURATION": request.duration_sec,
        "SEED": request.seed,
        "PROMPT": request.intent.visual_prompt,
        "MOTION_PROMPT": request.intent.motion_prompt,
        "NEGATIVE_PROMPT": request.intent.negative_prompt,
        "OUTPUT": request.output_path,
        "DEVICES": list(devices),
        "DEVICES_CSV": ",".join(devices),
        "CHARACTER_REFERENCE_PATHS": request.character_reference_images,
        "LOCATION_REFERENCE_PATHS": request.location_reference_images,
        "VOICE_REFERENCE_PATHS": request.voice_reference_files,
        "REFERENCE_PATHS": request.character_reference_images + request.location_reference_images,
        "CHARACTER_REFERENCE_DATA_URIS": char_uris,
        "LOCATION_REFERENCE_DATA_URIS": location_uris,
        "VOICE_REFERENCE_DATA_URIS": voice_uris,
        "REFERENCE_DATA_URIS": all_refs,
        "CONTINUATION_VIDEO_PATH": request.continuation_video_path,
        "CONTINUATION_FRAME_PATH": request.continuation_frame_path,
        "CONTINUATION_VIDEO_DATA_URI": continuation_video,
        "CONTINUATION_FRAME_DATA_URI": continuation_frame,
    }


async def download_file(url: str, destination: str | Path, *, timeout: float = 600.0) -> str:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=30.0), follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with destination.open("wb") as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)
    return str(destination)


def temporary_directory(prefix: str = "book2video_") -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(prefix=prefix)
