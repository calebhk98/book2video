from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from book2video.assembly import FFmpegAssembler
from book2video.config import ProjectConfig, load_config
from book2video.io_utils import choose_file_dialog, read_text_file
from book2video.pipeline import Book2VideoPipeline
from book2video.provider_loader import build_provider
from book2video.review import build_review_html, load_selected_and_takes, save_selection_override
from book2video.studio import serve_studio
from book2video.viewer import serve_viewer


def _load_chapter(
    args: argparse.Namespace,
    config: ProjectConfig,
    *,
    default_source: str | None = None,
) -> tuple[str, str]:
    chapter_path = getattr(args, "chapter_path", None)
    chapter_text = getattr(args, "chapter_text", None)
    pick_file = bool(getattr(args, "pick_file", False))

    if chapter_path:
        return read_text_file(chapter_path), Path(chapter_path).stem
    if chapter_text is not None:
        return chapter_text, "inline_chapter"
    if pick_file or default_source == "picker":
        picked = choose_file_dialog()
        if not picked:
            raise SystemExit("No file selected, or a desktop file picker was unavailable.")
        return read_text_file(picked), Path(picked).stem

    if default_source == "path":
        if not config.chapter_path or not str(config.chapter_path).strip():
            raise SystemExit('defaults.chapter_source = "path" requires chapter_path in config.toml.')
        return read_text_file(config.chapter_path), Path(config.chapter_path).stem

    if default_source == "text":
        if not config.chapter_text.strip():
            raise SystemExit('defaults.chapter_source = "text" requires non-empty chapter_text in config.toml.')
        return config.chapter_text, "chapter"

    if config.chapter_path and str(config.chapter_path).strip():
        return read_text_file(config.chapter_path), Path(config.chapter_path).stem
    return config.chapter_text, "chapter"


def _run_pipeline(
    config: ProjectConfig,
    chapter: str,
    name: str,
    *,
    llm_profile: str | None = None,
    video_profile: str | None = None,
):
    _, llm_cfg = config.resolve_llm_profile(llm_profile)
    _, video_cfg = config.resolve_video_profile(video_profile)
    llm = build_provider(llm_cfg.class_path, llm_cfg.options)
    video = build_provider(video_cfg.class_path, video_cfg.options)
    pipeline = Book2VideoPipeline(config, llm, video)
    return asyncio.run(pipeline.run(chapter, name))


def _latest_run(output_root: str | Path) -> Path:
    root = Path(output_root)
    if not root.exists():
        raise SystemExit(f"Output directory does not exist: {root}")
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / "08_selected.json").exists()]
    if not candidates:
        raise SystemExit(f"No completed runs found under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def cmd_run(args: argparse.Namespace) -> None:
    config_path = getattr(args, "run_config", None) or getattr(args, "config", "config.toml")
    config = load_config(config_path)
    chapter, name = _load_chapter(args, config)
    summary = _run_pipeline(
        config,
        chapter,
        name,
        llm_profile=getattr(args, "llm_profile", None),
        video_profile=getattr(args, "video_profile", None),
    )
    print(summary.model_dump_json(indent=2))


def cmd_select(args: argparse.Namespace) -> None:
    path = save_selection_override(args.run_dir, args.clip_id, args.take)
    print(f"Saved selection override: {path}")


def cmd_assemble(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    selected, takes = load_selected_and_takes(run_dir)
    output = FFmpegAssembler().assemble(selected, run_dir)
    review = build_review_html(run_dir, selected, takes)
    print(json.dumps({"rough_cut": output, "review": review}, indent=2))


def cmd_viewer(args: argparse.Namespace) -> None:
    config_path = getattr(args, "viewer_config", None) or getattr(args, "config", "config.toml")
    config = load_config(config_path)
    run_dir = Path(args.run_dir) if args.run_dir else _latest_run(config.output_root)
    serve_viewer(
        run_dir,
        host=args.host or config.defaults.viewer_host,
        port=args.port or config.defaults.viewer_port,
        open_browser=not args.no_browser,
        auto_assemble_on_selection=config.defaults.auto_assemble_on_selection,
        preload_adjacent=config.defaults.viewer_preload_adjacent,
    )


def cmd_studio(args: argparse.Namespace) -> None:
    config_path = getattr(args, "studio_config", None) or getattr(args, "config", "config.toml")
    config = load_config(config_path)
    serve_studio(
        config_path,
        host=args.host or config.defaults.viewer_host,
        port=args.port or config.defaults.viewer_port,
        open_browser=not args.no_browser,
    )


def cmd_default(config_path: str) -> None:
    config = load_config(config_path)
    defaults = config.defaults

    if defaults.action == "studio":
        serve_studio(
            config_path,
            host=defaults.viewer_host,
            port=defaults.viewer_port,
            open_browser=defaults.open_browser,
        )
        return

    if defaults.action == "run":
        args = argparse.Namespace(chapter_path=None, chapter_text=None, pick_file=False)
        chapter, name = _load_chapter(args, config, default_source=defaults.chapter_source)
        summary = _run_pipeline(config, chapter, name)
        print(summary.model_dump_json(indent=2))
        if defaults.launch_viewer_after_run:
            serve_viewer(
                summary.run_dir,
                host=defaults.viewer_host,
                port=defaults.viewer_port,
                open_browser=defaults.open_browser,
                auto_assemble_on_selection=defaults.auto_assemble_on_selection,
                preload_adjacent=defaults.viewer_preload_adjacent,
            )
        return

    run_dir = Path(defaults.viewer_run_dir) if defaults.viewer_run_dir else _latest_run(config.output_root)
    if defaults.action == "viewer":
        serve_viewer(
            run_dir,
            host=defaults.viewer_host,
            port=defaults.viewer_port,
            open_browser=defaults.open_browser,
            auto_assemble_on_selection=defaults.auto_assemble_on_selection,
            preload_adjacent=defaults.viewer_preload_adjacent,
        )
        return

    if defaults.action == "assemble":
        selected, takes = load_selected_and_takes(run_dir)
        output = FFmpegAssembler().assemble(selected, run_dir)
        review = build_review_html(run_dir, selected, takes)
        print(json.dumps({"rough_cut": output, "review": review}, indent=2))
        return

    raise SystemExit(f"Unknown default action: {defaults.action}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="book2video")
    parser.add_argument("--config", default="config.toml", help="Config used for no-subcommand/default behavior")
    sub = parser.add_subparsers(dest="command")

    studio = sub.add_parser("studio", help="Open the browser control center")
    studio.add_argument("--config", dest="studio_config", default=None)
    studio.add_argument("--host")
    studio.add_argument("--port", type=int)
    studio.add_argument("--no-browser", action="store_true")
    studio.set_defaults(func=cmd_studio)

    run = sub.add_parser("run", help="Plan, generate, judge, and assemble a chapter")
    run.add_argument("--config", dest="run_config", default=None)
    run.add_argument("--llm-profile", help="Named LLM profile; defaults to active_llm_profile")
    run.add_argument("--video-profile", help="Named video profile; defaults to active_video_profile")
    source = run.add_mutually_exclusive_group()
    source.add_argument("--chapter-path")
    source.add_argument("--chapter-text")
    source.add_argument("--pick-file", action="store_true")
    run.set_defaults(func=cmd_run)

    select = sub.add_parser("select", help="Override the selected take for one clip")
    select.add_argument("--run-dir", required=True)
    select.add_argument("--clip-id", required=True)
    select.add_argument("--take", required=True, type=int)
    select.set_defaults(func=cmd_select)

    assemble = sub.add_parser("assemble", help="Rebuild the rough cut after selection changes")
    assemble.add_argument("--run-dir", required=True)
    assemble.set_defaults(func=cmd_assemble)

    viewer = sub.add_parser("viewer", help="Open the standalone clip/take review viewer")
    viewer.add_argument("--config", dest="viewer_config", default=None)
    viewer.add_argument("--run-dir")
    viewer.add_argument("--host")
    viewer.add_argument("--port", type=int)
    viewer.add_argument("--no-browser", action="store_true")
    viewer.set_defaults(func=cmd_viewer)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        cmd_default(args.config)
        return
    args.func(args)


if __name__ == "__main__":
    main()
