"""Windows system-wide auto-paste (SendInput Ctrl+V) and soft audio cues."""

import ctypes
import math
import os
import struct
import sys
import time
from ctypes import wintypes

from .config import config_dir

INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002

VK_LSHIFT = 0xA0
VK_RSHIFT = 0xA1
VK_CONTROL = 0x11
VK_LCONTROL = 0xA2
VK_RCONTROL = 0xA3
VK_LMENU = 0xA4
VK_RMENU = 0xA5
VK_LWIN = 0x5B
VK_RWIN = 0x5C
VK_C = 0x43
VK_V = 0x56

_MODIFIER_VKS = (
    VK_LSHIFT, VK_RSHIFT,
    VK_LCONTROL, VK_RCONTROL,
    VK_LMENU, VK_RMENU,
    VK_LWIN, VK_RWIN,
)
_EXTENDED_VKS = {VK_RCONTROL, VK_RMENU, VK_LWIN, VK_RWIN}

_ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUT_UNION)]


if sys.platform == "win32":
    _u32 = ctypes.WinDLL("user32", use_last_error=True)
    _u32.GetForegroundWindow.restype = wintypes.HWND
    _u32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    _u32.GetWindowThreadProcessId.restype = wintypes.DWORD
    _u32.IsWindow.argtypes = [wintypes.HWND]
    _u32.IsWindow.restype = wintypes.BOOL
    _u32.SetForegroundWindow.argtypes = [wintypes.HWND]
    _u32.SetForegroundWindow.restype = wintypes.BOOL
    _u32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    _u32.GetAsyncKeyState.restype = ctypes.c_short
    _u32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
    _u32.MapVirtualKeyW.restype = wintypes.UINT
    _u32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    _u32.SendInput.restype = wintypes.UINT
    _u32.GetClipboardSequenceNumber.argtypes = []
    _u32.GetClipboardSequenceNumber.restype = wintypes.DWORD
else:
    _u32 = None


def _kbd_input(vk: int, key_up: bool = False) -> INPUT:
    flags = KEYEVENTF_KEYUP if key_up else 0
    if vk in _EXTENDED_VKS:
        flags |= KEYEVENTF_EXTENDEDKEY
    scan = _u32.MapVirtualKeyW(vk, 0) if _u32 else 0
    return INPUT(
        type=INPUT_KEYBOARD,
        ki=KEYBDINPUT(wVk=vk, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=0),
    )


def is_own_window(hwnd: int) -> bool:
    """True when `hwnd` belongs to the current Eloquent process."""
    if not hwnd or _u32 is None:
        return False
    pid = wintypes.DWORD(0)
    _u32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
    return pid.value == os.getpid()


def get_external_foreground() -> int:
    """Return current foreground HWND if it belongs to another process, else 0."""
    if _u32 is None:
        return 0
    hwnd = _u32.GetForegroundWindow()
    if hwnd and not is_own_window(int(hwnd)):
        return int(hwnd)
    return 0


def _held_modifiers() -> list[int]:
    if _u32 is None:
        return []
    return [vk for vk in _MODIFIER_VKS if _u32.GetAsyncKeyState(vk) & 0x8000]


def _wait_modifiers_release(timeout: float = 0.8) -> list[int]:
    deadline = time.monotonic() + timeout
    held = _held_modifiers()
    while held and time.monotonic() < deadline:
        time.sleep(0.015)
        held = _held_modifiers()
    return held


def _send_ctrl_combo(vk_key: int, target_hwnd: int = 0) -> bool:
    if _u32 is None:
        return False
    held = _wait_modifiers_release()
    cur_fg = _u32.GetForegroundWindow()
    if (not cur_fg or is_own_window(int(cur_fg))) and target_hwnd and _u32.IsWindow(wintypes.HWND(target_hwnd)):
        if not is_own_window(target_hwnd):
            _u32.SetForegroundWindow(wintypes.HWND(target_hwnd))
            time.sleep(0.05)

    # Flush any still-held modifiers separately first so Ctrl+C/V never merges into Ctrl+Shift+C
    if held:
        up_events = [_kbd_input(vk, key_up=True) for vk in held]
        up_arr = (INPUT * len(up_events))(*up_events)
        _u32.SendInput(len(up_events), up_arr, ctypes.sizeof(INPUT))
        time.sleep(0.025)

    events = [
        _kbd_input(VK_CONTROL, key_up=False),
        _kbd_input(vk_key, key_up=False),
        _kbd_input(vk_key, key_up=True),
        _kbd_input(VK_CONTROL, key_up=True),
    ]
    arr = (INPUT * len(events))(*events)
    sent = _u32.SendInput(len(events), arr, ctypes.sizeof(INPUT))
    return sent == len(events)


def send_paste(target_hwnd: int = 0) -> bool:
    """Wait for shortcut modifiers to clear, restore focus only if needed, and send Ctrl+V."""
    return _send_ctrl_combo(VK_V, target_hwnd=target_hwnd)


def capture_selection_copy(target_hwnd: int = 0, wait_sec: float = 0.14) -> bool:
    """Send Ctrl+C after releasing hotkey modifiers; return True if clipboard updated."""
    if _u32 is None:
        return False
    seq_before = _u32.GetClipboardSequenceNumber()
    if not _send_ctrl_combo(VK_C, target_hwnd=target_hwnd):
        return False
    deadline = time.monotonic() + wait_sec
    while time.monotonic() < deadline:
        if _u32.GetClipboardSequenceNumber() != seq_before:
            return True
        time.sleep(0.015)
    return _u32.GetClipboardSequenceNumber() != seq_before


# --------------------------------------------------------------------------- #
# subtle async audio cues (cached PCM WAV files via SND_FILENAME | SND_ASYNC)
# --------------------------------------------------------------------------- #
_CUE_VER = "v2"
_CUE_SPECS = {
    "start": ((587.33, 80), (783.99, 105)),
    "done": ((783.99, 75), (1046.50, 120)),
    "cancel": ((523.25, 75), (392.00, 100)),
    "error": ((349.23, 85), (293.66, 110)),
}
_CUE_FILES: dict[str, str] = {}


def _synth_wav(notes: tuple[tuple[float, int], ...], amplitude: float = 0.28, rate: int = 22050) -> bytes:
    # 35ms silence lead-in wakes idle Windows/Realtek DACs so the first note isn't clipped
    samples = bytearray(b"\x00\x00" * int(rate * 0.035))
    gap = b"\x00\x00" * int(rate * 0.015)
    fade_n = max(1, int(rate * 0.014))
    for idx, (freq, dur_ms) in enumerate(notes):
        if idx > 0:
            samples.extend(gap)
        n = max(1, int(rate * dur_ms / 1000.0))
        step = 2.0 * math.pi * freq / rate
        for i in range(n):
            env = 1.0
            if i < fade_n:
                env = 0.5 * (1.0 - math.cos(math.pi * i / fade_n))
            elif i >= n - fade_n:
                rem = max(0, n - 1 - i)
                env = 0.5 * (1.0 - math.cos(math.pi * rem / fade_n))
            val = int(32767 * amplitude * env * math.sin(step * i))
            samples.extend(struct.pack("<h", val))
    data = bytes(samples)
    hdr = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data), b"WAVE",
        b"fmt ", 16, 1, 1, rate, rate * 2, 2, 16,
        b"data", len(data),
    )
    return hdr + data


def _ensure_cue_file(kind: str) -> str:
    path = _CUE_FILES.get(kind)
    if path and os.path.exists(path):
        return path
    spec = _CUE_SPECS.get(kind)
    if spec is None:
        return ""
    cue_dir = config_dir() / "cues"
    cue_dir.mkdir(parents=True, exist_ok=True)
    target = cue_dir / f"{kind}_{_CUE_VER}.wav"
    if not target.exists():
        target.write_bytes(_synth_wav(spec))
    path = str(target)
    _CUE_FILES[kind] = path
    return path


def play_cue(kind: str) -> None:
    """Play a gentle non-blocking sound cue on Windows."""
    if sys.platform != "win32":
        return
    path = _ensure_cue_file(kind)
    if not path:
        return
    try:
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
    except Exception:
        pass
