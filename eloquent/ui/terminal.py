"""Terminal UI (Rich).

Finished paragraphs scroll up permanently (polished text, fixes underlined).
A small live area at the bottom shows what is still in flight:
sentences waiting for cleanup, the paragraph being polished and the grey live partial.

Keys: Space = start/stop, Q = quit (Ctrl+C also works).
"""

import queue
import sys
import threading
import time

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

try:
    import msvcrt  # Windows: non-blocking key reads
except ImportError:  # pragma: no cover
    msvcrt = None

MAX_PENDING_LINES = 6


def _paragraph_text(p: dict) -> Text:
    t = Text()
    t.append(p["time"] + "  ", style="dim")
    if p["tokens"]:
        for i, (w, changed) in enumerate(p["tokens"]):
            if i:
                t.append(" ")
            t.append(w, style="underline yellow" if changed else "")
    else:
        t.append(p["raw"])
    if p["note"]:
        t.append(f"\n       {p['note']}", style="dim yellow")
    return t


class TerminalUI:
    def __init__(self, engine, wav_path=None, auto_start=True):
        self.engine = engine
        self.wav_path = wav_path
        self.auto_start = auto_start
        self.console = Console(highlight=False)
        self.changed = threading.Event()
        self.logs: "queue.Queue[tuple]" = queue.Queue()
        self.printed = 0           # number of paragraphs already printed permanently
        self.quit = False
        self.ended = threading.Event()
        self.last_seg_info = ""
        engine.subscribe(self._on_event)

    # -- events (any thread) ----------------------------------------------
    def _on_event(self, kind, data):
        if kind == "log":
            self.logs.put(data)
        elif kind == "segment":
            self.last_seg_info = f"{data['audio_sec']:.1f}s audio decoded in {data['decode_sec']:.2f}s"
        elif kind == "reset":
            self.printed = 0
        elif kind == "session_started":
            self.logs.put(("info", f"Recording... saving to {data['dir']}"))
        elif kind == "session_ended":
            self.logs.put(("info", f"Stopped. Files in: {data['dir']}"))
            self.ended.set()
        self.changed.set()

    # -- rendering ----------------------------------------------------------
    def _flush_logs(self, live):
        styles = {"info": "dim", "warn": "yellow", "error": "bold red"}
        while True:
            try:
                level, msg = self.logs.get_nowait()
            except queue.Empty:
                return
            live.console.print(Text(msg, style=styles.get(level, "")))

    def _print_finished(self, live, snap):
        paras = snap["paragraphs"]
        while self.printed < len(paras) and paras[self.printed]["status"] != "cleaning":
            live.console.print(_paragraph_text(paras[self.printed]))
            live.console.print()
            self.printed += 1

    def _status_line(self) -> Text:
        e = self.engine
        colors = {"listening": "green", "stopping": "yellow", "idle": "bright_black", "loading": "cyan"}
        t = Text()
        t.append("● ", style=colors.get(e.status, ""))
        t.append(e.status, style="bold")
        llm = e.llm
        llm_txt = {"ready": llm["label"], "checking": "checking " + llm["label"],
                   "error": "LLM error (raw kept)", "off": "LLM off"}.get(llm["state"], "")
        t.append(f"   {llm_txt}", style="dim")
        if self.last_seg_info:
            t.append(f"   last: {self.last_seg_info}", style="dim")
        hint = "  Space start/stop · Q quit" if msvcrt and not self.wav_path else "  Ctrl+C stop"
        t.append(hint, style="dim")
        return t

    def _live_view(self, snap) -> Group:
        lines = []
        for p in snap["paragraphs"][self.printed:]:
            t = Text(p["time"] + "  ", style="dim")
            t.append(p["raw"], style="italic bright_black")
            t.append("  polishing...", style="cyan")
            lines.append(t)
        for x in snap["loose"]:
            t = Text(x["t"] + "  ", style="dim")
            t.append(x["text"])
            lines.append(t)
        if snap["partial"]:
            lines.append(Text("…  " + snap["partial"], style="italic bright_black"))
        lines = lines[-MAX_PENDING_LINES:]
        lines.append(self._status_line())
        return Group(*lines)

    # -- keys -------------------------------------------------------------
    def _keys(self):
        if not msvcrt or self.wav_path:
            return
        while msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch == " ":
                self.engine.toggle()
            elif ch.lower() == "q":
                self.quit = True

    # -- main loop ----------------------------------------------------------
    def run(self):
        e = self.engine
        err = []

        def load():
            try:
                e.load()
            except Exception as ex:
                err.append(ex)

        loader = threading.Thread(target=load, daemon=True)
        loader.start()
        with self.console.status("[cyan]Loading speech model...", spinner="dots"):
            while loader.is_alive():
                loader.join(0.1)
                while not self.logs.empty():
                    self.console.print(Text(self.logs.get()[1], style="dim"))
        if err:
            raise err[0]
        if self.auto_start or self.wav_path:
            e.start(self.wav_path)
        try:
            with Live(self._live_view(e.state.snapshot()), console=self.console,
                      auto_refresh=False, transient=False, vertical_overflow="visible") as live:
                while True:
                    self.changed.wait(0.1)
                    self.changed.clear()
                    self._keys()
                    snap = e.state.snapshot()
                    self._flush_logs(live)
                    self._print_finished(live, snap)
                    live.update(self._live_view(snap), refresh=True)
                    if self.quit or (self.wav_path and self.ended.is_set()):
                        break
                self._shutdown(live)
        except KeyboardInterrupt:
            self.console.print("[dim]Stopping... finishing the last sentence and cleanup[/dim]")
            e.stop()
            e.wait_idle(40)
            snap = e.state.snapshot()
            for p in snap["paragraphs"][self.printed:]:
                self.console.print(_paragraph_text(p))
            while not self.logs.empty():
                self.console.print(Text(self.logs.get()[1], style="dim"))

    def _shutdown(self, live):
        e = self.engine
        if e.busy:
            live.console.print(Text("Stopping... finishing the last sentence and cleanup", style="dim"))
            e.stop()
            deadline = time.time() + 40
            while e.busy and time.time() < deadline:
                time.sleep(0.1)
                snap = e.state.snapshot()
                self._flush_logs(live)
                self._print_finished(live, snap)
                live.update(self._live_view(snap), refresh=True)
        snap = e.state.snapshot()
        self._flush_logs(live)
        self._print_finished(live, snap)
        live.update(self._status_line(), refresh=True)


def run_terminal(engine, wav_path=None, auto_start=True):
    TerminalUI(engine, wav_path, auto_start).run()
    sys.stdout.flush()
