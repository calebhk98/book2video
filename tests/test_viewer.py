from pathlib import Path

from fastapi.testclient import TestClient

from book2video.io_utils import write_json
from book2video.schemas import GeneratedTake, SelectedClip
from book2video.viewer import create_viewer_app, viewer_html


def _make_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    take_dir = run / "takes" / "clip_001"
    take_dir.mkdir(parents=True)
    a = take_dir / "t01.mp4"
    b = take_dir / "t02.mp4"
    a.write_bytes(b"not-a-real-video")
    b.write_bytes(b"not-a-real-video")
    selected = [
        SelectedClip(
            clip_id="clip_001",
            scene_id="scene_001",
            shot_id="shot_001",
            selected_take=1,
            selected_path=str(a),
        )
    ]
    takes = {
        "clip_001": [
            GeneratedTake(clip_id="clip_001", take_number=1, output_path=str(a), provider_id="mock"),
            GeneratedTake(clip_id="clip_001", take_number=2, output_path=str(b), provider_id="mock"),
        ]
    }
    write_json(run / "06_clips.json", [{"clip_id": "clip_001", "target_duration_sec": 5.0}])
    write_json(run / "08_selected.json", [x.model_dump(mode="json") for x in selected])
    write_json(run / "09_all_takes.json", {k: [x.model_dump(mode="json") for x in v] for k, v in takes.items()})
    return run


def test_viewer_can_change_any_take(tmp_path: Path):
    run = _make_run(tmp_path)
    client = TestClient(create_viewer_app(run))
    state = client.get("/api/state").json()
    assert state["clips"][0]["selected_take"] == 1

    response = client.post("/api/select", json={"clip_id": "clip_001", "take_number": 2})
    assert response.status_code == 200
    assert response.json()["clips"][0]["selected_take"] == 2
    assert (run / "selection_overrides.json").exists()


def test_viewer_preloads_three_slots_and_restarts_changed_current_take():
    html = viewer_html(preload_adjacent=True)
    assert html.count('class="videoSlot"') == 3
    assert 'const PRELOAD_ADJACENT = true;' in html
    assert 'activateIndex(currentIndex, shouldResume, true);' in html
    assert 'slot.el.currentTime = 0' in html


def test_viewer_can_disable_adjacent_preload():
    html = viewer_html(preload_adjacent=False)
    assert 'const PRELOAD_ADJACENT = false;' in html
