from __future__ import annotations

import html
import json
from pathlib import Path

from book2video.io_utils import read_json
from book2video.schemas import GeneratedTake, SelectedClip



def load_selected_and_takes(run_dir: str | Path) -> tuple[list[SelectedClip], dict[str, list[GeneratedTake]]]:
    """Load generated choices and apply any human selection overrides without rewriting the original AI decision."""
    run_dir = Path(run_dir)
    selected = [SelectedClip.model_validate(x) for x in read_json(run_dir / "08_selected.json")]
    take_data = read_json(run_dir / "09_all_takes.json")
    takes = {k: [GeneratedTake.model_validate(x) for x in values] for k, values in take_data.items()}
    overrides_path = run_dir / "selection_overrides.json"
    overrides = read_json(overrides_path) if overrides_path.exists() else {}
    for clip in selected:
        if clip.clip_id in overrides:
            wanted = int(overrides[clip.clip_id])
            replacement = next((x for x in takes.get(clip.clip_id, []) if x.take_number == wanted), None)
            if replacement:
                clip.selected_take = wanted
                clip.selected_path = replacement.output_path
    return selected, takes

def build_review_html(
    run_dir: str | Path,
    selected: list[SelectedClip],
    all_takes: dict[str, list[GeneratedTake]],
) -> str:
    run_dir = Path(run_dir)
    rows: list[str] = []
    for clip in selected:
        takes = all_takes.get(clip.clip_id, [])
        cards = []
        for take in takes:
            try:
                rel = Path(take.output_path).resolve().relative_to(run_dir.resolve())
                src = rel.as_posix()
            except ValueError:
                src = Path(take.output_path).as_uri() if Path(take.output_path).is_absolute() else take.output_path
            marker = " SELECTED" if take.take_number == clip.selected_take else ""
            cards.append(
                f"<div class='take'><h4>Take {take.take_number}{marker}</h4>"
                f"<video controls preload='metadata' src='{html.escape(src)}'></video></div>"
            )
        issue_text = ""
        if clip.judgement:
            issue_text = html.escape(clip.judgement.reason or "")
        rows.append(
            f"<section id='{html.escape(clip.clip_id)}'>"
            f"<h2>{html.escape(clip.clip_id)} — selected take {clip.selected_take}</h2>"
            f"<p>{issue_text}</p><div class='takes'>{''.join(cards)}</div></section>"
        )

    page = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Book2Video review</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem}}
section{{border-top:1px solid #bbb;padding:1rem 0 2rem}}
.takes{{display:flex;gap:1rem;flex-wrap:wrap}} .take{{width:31%;min-width:300px}}
video{{width:100%;background:#111}} code{{white-space:pre-wrap}}
</style></head><body>
<h1>Rough-cut take review</h1>
<p>Selection changes can be made with the CLI: <code>book2video select --run-dir ... --clip-id CLIP --take N</code>.</p>
{''.join(rows)}
</body></html>"""
    path = run_dir / "review.html"
    path.write_text(page, encoding="utf-8")
    return str(path)


def save_selection_override(run_dir: str | Path, clip_id: str, take_number: int) -> str:
    path = Path(run_dir) / "selection_overrides.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[clip_id] = take_number
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path)
