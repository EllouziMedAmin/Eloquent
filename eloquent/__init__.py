"""Eloquent - open-source live transcription for Windows (CPU friendly).

Parakeet TDT 0.6B v3 (INT8, sherpa-onnx) + Silero VAD, with optional
background LLM proofreading (Gemini 3.1 Flash-Lite by default).
"""

import ctypes
import os
import sys

APP_NAME = "Eloquent"
__version__ = "0.1.0"

if sys.platform == "win32":
    # Preload Windows System32 ICU DLLs before any Qt/PySide6 import so Anaconda's
    # older ICU DLLs in Library\bin don't shadow them on Conda-based Python installs.
    _sys32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    for _dll in ("icudt.dll", "icuuc.dll", "icuin.dll"):
        _p = os.path.join(_sys32, _dll)
        if os.path.exists(_p):
            try:
                ctypes.WinDLL(_p)
            except OSError:
                pass

