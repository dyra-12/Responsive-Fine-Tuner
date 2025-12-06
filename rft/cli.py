"""Command-line interface for Responsive Fine-Tuner (rft).

This CLI keeps imports lazy so `rft --help` works even when heavy ML
dependencies are not installed. The `demo` command attempts to launch the
bundled demo using the project's `run_app.py` entry.
"""

import argparse
import os
import sys


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_cwd_is_project_root():
    root = _project_root()
    os.chdir(root)


def cmd_demo(args: argparse.Namespace):
    """Run the demo app using the repo's `run_app.py`.

    This imports `run_app` lazily and forwards CLI-style args.
    If required dependencies are missing, a friendly message is printed.
    """
    _ensure_cwd_is_project_root()

    # Prepare argv for run_app.main
    run_argv = ["run_app.py"]
    if args.port:
        run_argv.extend(["--port", str(args.port)])
    if args.share:
        run_argv.append("--share")
    if args.debug:
        run_argv.append("--debug")
    if args.config:
        run_argv.extend(["--config", args.config])

    try:
        # Import as late as possible to avoid heavy dependency errors on --help
        from run_app import main as run_app_main
    except Exception as e:  # pragma: no cover - user environment may differ
        print("\nCould not import RFT demo runner. This usually means required\n" \
              "dependencies (gradio/transformers/torch) are not installed.\n\n" \
              "To try the full demo, install requirements: `pip install -r requirements.txt`.\n")
        print("Import error:", e)
        sys.exit(2)

    # Patch sys.argv for run_app.main to parse
    old_argv = sys.argv
    try:
        sys.argv = run_argv
        run_app_main()
    finally:
        sys.argv = old_argv


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="rft", description="Responsive Fine-Tuner (RFT) CLI")
    sub = parser.add_subparsers(dest="command")

    p_demo = sub.add_parser("demo", help="Launch the demo app with example data")
    p_demo.add_argument("--port", type=int, default=7860, help="Port to run the demo on")
    p_demo.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    p_demo.add_argument("--debug", action="store_true", help="Enable debug mode")
    p_demo.add_argument("--config", type=str, default="config/settings.yaml", help="Path to config file")

    p_run = sub.add_parser("run", help="Alias for demo; run the app with provided args")
    p_run.add_argument("--port", type=int, default=7860)
    p_run.add_argument("--config", type=str, default="config/settings.yaml")

    p_info = sub.add_parser("info", help="Show project paths and quick helper info")

    args = parser.parse_args(argv)

    if args.command in ("demo", "run"):
        # Use demo behavior for run as well
        cmd_demo(args)
    elif args.command == "info":
        _ensure_cwd_is_project_root()
        print("Project root:", _project_root())
        print("Example data:", os.path.join(_project_root(), "data", "example"))
        print("Start demo: `rft demo --port 7860` or `python run_app.py --port 7860`")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
