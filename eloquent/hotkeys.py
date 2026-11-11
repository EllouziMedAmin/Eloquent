"""Global hotkeys on Windows: key parsing + the tiny always-on launcher.

The launcher (`python -m eloquent --listen`) imports neither Qt nor the speech
models, so it costs a few MB. It only waits for the launch hotkey and then
starts (or brings forward) the Eloquent widget.
"""

import ctypes
import subprocess
import sys
from ctypes import wintypes
from pathlib import Path

from .config import ROOT, config_dir

WM_HOTKEY = 0x0312
ERROR_ALREADY_EXISTS = 183

_MODS = {"ctrl": 0x2, "control": 0x2, "alt": 0x1, "shift": 0x4, "win": 0x8}
_KEYS = {
    "space": 0x20, "tab": 0x09, "enter": 0x0D, "return": 0x0D, "esc": 0x1B, "escape": 0x1B,
    "pause": 0x13, "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pgup": 0x21, "pagedown": 0x22, "pgdn": 0x22,
    "insert": 0x2D, "ins": 0x2D, "delete": 0x2E, "del": 0x2E, "backspace": 0x08,
}


def parse_hotkey(text: str):
    """'ctrl+alt+space' or 'ctrl+up' -> (modifiers, virtual key) or None."""
    parts = [p.strip().lower() for p in text.split("+") if p.strip()]
    mods, vk = 0, None
    for p in parts:
        if p in _MODS:
            mods |= _MODS[p]
        elif p in _KEYS:
            vk = _KEYS[p]
        elif len(p) == 1 and p.isalnum():
            vk = ord(p.upper())
        elif p[0] == "f" and p[1:].isdigit() and 1 <= int(p[1:]) <= 24:
            vk = 0x70 + int(p[1:]) - 1
        else:
            return None
    if vk is None or mods == 0:
        return None
    return mods | 0x4000, vk  # MOD_NOREPEAT


def single_instance(name: str) -> bool:
    """True if this process is the first one holding the named mutex."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.restype = wintypes.HANDLE
    handle = k32.CreateMutexW(None, False, name)
    single_instance.handles.append(handle)  # keep it open for the life of the process
    return ctypes.get_last_error() != ERROR_ALREADY_EXISTS


single_instance.handles = []


def _pythonw() -> str:
    exe = Path(sys.executable)
    pw = exe.with_name("pythonw.exe")
    return str(pw if pw.exists() else exe)


def launch_widget():
    DETACHED_PROCESS, CREATE_NO_WINDOW = 0x00000008, 0x08000000
    subprocess.Popen([_pythonw(), "-m", "eloquent", "--ui", "widget"], cwd=str(ROOT),
                     creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW, close_fds=True)


def spawn_listener():
    """Start the launcher in the background (no-op if one is already running)."""
    DETACHED_PROCESS, CREATE_NO_WINDOW = 0x00000008, 0x08000000
    subprocess.Popen([_pythonw(), "-m", "eloquent", "--listen"], cwd=str(ROOT),
                     creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW, close_fds=True)


def run_listener(hotkey: str) -> int:
    if not single_instance("Eloquent.HotkeyListener"):
        return 0
    log = config_dir() / "hotkey.log"
    parsed = parse_hotkey(hotkey)
    if not parsed:
        log.write_text(f"Invalid launch hotkey: {hotkey!r}\n", encoding="utf-8")
        return 1
    u32 = ctypes.windll.user32
    if not u32.RegisterHotKey(None, 1, parsed[0], parsed[1]):
        log.write_text(f"Could not register {hotkey!r}: another app already uses it.\n", encoding="utf-8")
        return 1
    log.write_text(f"Listening for {hotkey}\n", encoding="utf-8")
    msg = wintypes.MSG()
    while u32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
        if msg.message == WM_HOTKEY:
            launch_widget()
    return 0
