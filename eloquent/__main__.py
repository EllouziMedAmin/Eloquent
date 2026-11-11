"""Eloquent command line.

    python -m eloquent                    # interface from settings (widget by default)
    python -m eloquent --ui terminal
    python -m eloquent --ui web
    python -m eloquent --input talk.wav   # transcribe a recording through the same pipeline

Command-line options apply to this run only; use the widget settings (or edit
%APPDATA%\\Eloquent\\settings.json) to change the defaults.
"""

import argparse
import os
import sys

from . import APP_NAME, __version__
from .config import SETTINGS_FILE, Settings
from .llm.prompts import REPORT_FORMATS


def build_parser(s: Settings) -> argparse.ArgumentParser:
    fmt_ids = [f["id"] for f in REPORT_FORMATS]
    p = argparse.ArgumentParser(prog="eloquent", description=f"{APP_NAME} {__version__} - live transcription on CPU")
    p.add_argument("--ui", choices=["widget", "terminal", "web"], default=None,
                   help=f"interface (default from settings: {s.ui})")
    p.add_argument("--input", metavar="WAV", help="transcribe a WAV file instead of the microphone")
    p.add_argument("--format", choices=fmt_ids, default=None,
                   help="generate a formatted document at the end of the session (summary, bullets, meeting, lecture, report, article)")
    p.add_argument("--prompt", default="", help="custom LLM instruction to run at the end of the session")
    p.add_argument("--no-start", action="store_true", help="terminal/web: wait for Space / Start instead of recording at once")
    audio = p.add_mutually_exclusive_group()
    audio.add_argument("--save-audio", dest="save_audio", action="store_true", default=None,
                       help="save audio.wav in the session folder")
    audio.add_argument("--no-audio", dest="save_audio", action="store_false", help="do not save audio.wav")
    p.add_argument("--no-clean", action="store_true", help="skip LLM cleanup, raw text only")
    p.add_argument("--model", help=f"LLM model (default {s.llm_model})")
    p.add_argument("--keep-fillers", action="store_true", default=None, help="keep uh/um in the cleaned text")
    p.add_argument("--threads", type=int, help="CPU threads for Parakeet (default: half the cores, max 8)")
    p.add_argument("--device", type=int, help="microphone index (see --list-devices)")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--silence", type=float, help=f"pause (s) that ends a sentence (default {s.silence})")
    p.add_argument("--out", help="sessions folder")
    p.add_argument("--port", type=int, help=f"web UI port (default {s.web_port})")
    p.add_argument("--no-browser", action="store_true", help="web: do not open the browser automatically")
    p.add_argument("--minimized", action="store_true", help="widget: start minimized in background")
    p.add_argument("--create-shortcut", action="store_true", help="create Desktop/Start Menu/Startup shortcuts and start the hotkey launcher")
    p.add_argument("--listen", action="store_true", help="run only the tiny background launcher that starts Eloquent on the launch hotkey")
    p.add_argument("--settings-path", action="store_true", help="print where settings are stored")
    p.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    return p


def main(argv=None):
    s = Settings.load()
    args = build_parser(s).parse_args(argv)

    if args.settings_path:
        print(SETTINGS_FILE)
        return 0
    if args.create_shortcut:
        from .hotkeys import spawn_listener
        from .shortcuts import create_windows_shortcuts
        for p in create_windows_shortcuts():
            print(f"Created shortcut: {p}")
        spawn_listener()
        print(f"Launcher running in background: press {s.launch_hotkey} to start Eloquent "
              f"(it also starts with Windows).")
        return 0
    if args.listen:
        from .hotkeys import run_listener
        return run_listener(s.launch_hotkey)
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    # per-run overrides (never written back to settings.json)
    if args.save_audio is not None:
        s.save_audio = args.save_audio
    if args.no_clean:
        s.cleanup = False
    if args.model:
        s.llm_model = args.model
    if args.keep_fillers:
        s.keep_fillers = True
    if args.threads:
        s.threads = args.threads
    if args.device is not None:
        s.device = args.device
    if args.silence:
        s.silence = args.silence
    if args.out:
        s.sessions_dir = args.out
    if args.port:
        s.web_port = args.port

    ui = args.ui or ("terminal" if args.input else s.ui)
    if sys.stdout is not None and hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        os.system("")  # enable ANSI colours in the Windows console

    from .engine import Engine
    engine = Engine(s)

    if ui == "widget":
        try:
            from .ui.widget import run_widget
        except ImportError as e:
            print(f"Widget UI unavailable ({e}). Install it with: pip install PySide6-Essentials\n"
                  f"Falling back to the terminal UI.")
            ui = "terminal"
        else:
            return run_widget(engine, s, start_minimized=args.minimized)

    if ui == "web":
        from .ui.web import run_web
        run_web(engine, s.web_port, open_browser=not args.no_browser, wav_path=args.input)
        return 0

    from .ui.terminal import run_terminal
    run_terminal(engine, wav_path=args.input, auto_start=not args.no_start,
                 auto_format=args.format or ("custom" if args.prompt else None),
                 custom_prompt=args.prompt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
