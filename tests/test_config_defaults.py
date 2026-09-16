from pathlib import Path

from book2video.config import load_config


def test_defaults_are_optional(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        '''
chapter_text = "hello"
[llm]
class_path = "book2video.providers.mock_llm:MockLLMProvider"
[video]
class_path = "book2video.providers.mock_video:MockVideoProvider"
''',
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.defaults.action == "studio"
    assert config.defaults.chapter_source == "config"
    assert config.defaults.viewer_port == 8765
    assert config.defaults.viewer_preload_adjacent is True


def test_defaults_can_choose_picker_and_viewer(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        '''
chapter_text = ""
[defaults]
action = "viewer"
chapter_source = "picker"
viewer_port = 9999
viewer_preload_adjacent = false
[llm]
class_path = "book2video.providers.mock_llm:MockLLMProvider"
[video]
class_path = "book2video.providers.mock_video:MockVideoProvider"
''',
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.defaults.action == "viewer"
    assert config.defaults.chapter_source == "picker"
    assert config.defaults.viewer_port == 9999
    assert config.defaults.viewer_preload_adjacent is False


def test_defaults_accept_explicit_path_and_text_sources(tmp_path: Path):
    for source in ("path", "text"):
        path = tmp_path / f"{source}.toml"
        path.write_text(
            f'''
chapter_text = "hello"
chapter_path = "./chapter.md"
[defaults]
chapter_source = "{source}"
[llm]
class_path = "book2video.providers.mock_llm:MockLLMProvider"
[video]
class_path = "book2video.providers.mock_video:MockVideoProvider"
''',
            encoding="utf-8",
        )
        assert load_config(path).defaults.chapter_source == source


def test_named_profiles_resolve_without_removing_legacy_support(tmp_path: Path):
    path = tmp_path / "profiles.toml"
    path.write_text(
        '''
active_llm_profile = "local"
active_video_profile = "fast"
[llm_profiles.local]
class_path = "book2video.providers.mock_llm:MockLLMProvider"
label = "Local"
[video_profiles.fast]
class_path = "book2video.providers.mock_video:MockVideoProvider"
label = "Fast"
''',
        encoding="utf-8",
    )
    config = load_config(path)
    llm_name, llm = config.resolve_llm_profile()
    video_name, video = config.resolve_video_profile()
    assert llm_name == "local"
    assert llm.label == "Local"
    assert video_name == "fast"
    assert video.label == "Fast"


def test_legacy_worker_groups_imply_manual_device_mode(tmp_path: Path):
    path = tmp_path / "legacy-gpus.toml"
    path.write_text(
        '''
chapter_text = "hello"
[generation]
worker_device_groups = [["cuda:0"], ["cuda:1"]]
[llm]
class_path = "book2video.providers.mock_llm:MockLLMProvider"
[video]
class_path = "book2video.providers.mock_video:MockVideoProvider"
''',
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.generation.device_mode == "manual"
    assert config.generation.worker_device_groups == [["cuda:0"], ["cuda:1"]]
