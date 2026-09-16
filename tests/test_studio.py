from pathlib import Path

from fastapi.testclient import TestClient

from book2video.studio import STUDIO_HTML, create_studio_app


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(
        f'''
chapter_text = "hello"
output_root = "{(tmp_path / 'out').as_posix()}"
active_llm_profile = "mock"
active_video_profile = "mock"
[defaults]
action = "studio"
viewer_preload_adjacent = true
[llm_profiles.mock]
label = "Mock LLM"
description = "test llm"
class_path = "book2video.providers.mock_llm:MockLLMProvider"
[video_profiles.mock]
label = "Mock video"
description = "test video"
class_path = "book2video.providers.mock_video:MockVideoProvider"
[video_profiles.mock.options]
max_duration_sec = 10.0
''',
        encoding="utf-8",
    )
    return path


def test_studio_bootstrap_exposes_profiles_and_browser_controls(tmp_path: Path):
    config = _config(tmp_path)
    client = TestClient(create_studio_app(config))
    data = client.get("/api/bootstrap").json()
    assert data["defaults"]["llm_profile"] == "mock"
    assert data["defaults"]["video_profile"] == "mock"
    assert data["llm_profiles"]["mock"]["label"] == "Mock LLM"
    assert 'id="chapterSource"' in STUDIO_HTML
    assert 'id="llmProfile"' in STUDIO_HTML
    assert 'id="videoProfile"' in STUDIO_HTML
    assert 'title="' in STUDIO_HTML
    assert 'id="rslot0"' in STUDIO_HTML and 'id="rslot2"' in STUDIO_HTML
    assert 'id="deviceMode"' in STUDIO_HTML
    assert 'id="hardwareList"' in STUDIO_HTML
    assert 'id="statVideo"' in STUDIO_HTML


def test_browser_created_profile_overlays_toml_profile_without_editing_toml(tmp_path: Path):
    config = _config(tmp_path)
    original = config.read_text(encoding="utf-8")
    client = TestClient(create_studio_app(config))
    response = client.post(
        "/api/profile",
        json={
            "kind": "video",
            "name": "wan_local",
            "class_path": "book2video.providers.command_video:CommandVideoProvider",
            "label": "Wan local",
            "description": "test overlay",
            "options": {"command": ["true"], "max_duration_sec": 10.0},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["video_profiles"]["wan_local"]["custom"] is True
    assert config.read_text(encoding="utf-8") == original
    assert config.with_name("config.studio.json").exists()
