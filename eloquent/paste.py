"""Windows system-wide auto-paste (SendInput Ctrl+V) and soft audio cues."""

import ctypes
import math
import os
import struct
import sys
import time
from ctypes import wintypes

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


def _wait_modifiers_release(timeout: float = 0.5) -> list[int]:
    deadline = time.monotonic() + timeout
    held = _held_modifiers()
    while held and time.monotonic() < deadline:
        time.sleep(0.015)
        held = _held_modifiers()
    return held


def send_paste(target_hwnd: int = 0) -> bool:
    """Wait for shortcut modifiers to clear, restore focus only if needed, and send Ctrl+V."""
    if _u32 is None:
        return False
    held = _wait_modifiers_release()
    cur_fg = _u32.GetForegroundWindow()
    if cur_fg and is_own_window(int(cur_fg)) and target_hwnd and _u32.IsWindow(wintypes.HWND(target_hwnd)):
        if not is_own_window(target_hwnd):
            _u32.SetForegroundWindow(wintypes.HWND(target_hwnd))
            time.sleep(0.05)

    events = [_kbd_input(vk, key_up=True) for vk in held]
    events.extend([
        _kbd_input(VK_CONTROL, key_up=False),
        _kbd_input(VK_V, key_up=False),
        _kbd_input(VK_V, key_up=True),
        _kbd_input(VK_CONTROL, key_up=True),
    ])
    arr = (INPUT * len(events))(*events)
    sent = _u32.SendInput(len(events), arr, ctypes.sizeof(INPUT))
    return sent == len(events)


# --------------------------------------------------------------------------- #
# subtle async audio cues (pure stdlib PCM WAV kept alive for SND_ASYNC)
# --------------------------------------------------------------------------- #
_CUE_SPECS = {
    "start": ((587.33, 55), (783.99, 75)),
    "done": ((783.99, 50), (1046.50, 90)),
    "cancel": ((523.25, 55), (392.00, 75)),
    "error": ((349.23, 65), (293.66, 85)),
}
_CUES: dict[str, bytes] = {}


def _synth_wav(notes: tuple[tuple[float, int], ...], amplitude: float = 0.14, rate: int = 22050) -> bytes:
    samples = bytearray()
    gap = b"\x00\x00" * int(rate * 0.012)
    fade_n = max(1, int(rate * 0.012))
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


def play_cue(kind: str) -> None:
    """Play a gentle non-blocking sound cue on Windows."""
    if sys.platform != "win32":
        return
    spec = _CUE_SPECS.get(kind)
    if spec is None:
        return
    wav = _CUES.get(kind)
    if wav is None:
        wav = _synth_wav(spec)
        _CUES[kind] = wav
    try:
        import winsound
        winsound.PlaySound(wav, winsound.SND_MEMORY | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
    except Exception:
        pass
