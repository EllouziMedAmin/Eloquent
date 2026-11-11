"""Windows Desktop / Start Menu / Startup shortcuts for Eloquent.

The launch hotkey is NOT stored in the .lnk files (Explorer applies those
unreliably). A tiny background launcher (`eloquent --listen`) owns it instead,
and starts with Windows through the Startup shortcut.
"""

import os
import subprocess
import sys
from pathlib import Path

from . import APP_NAME
from .config import ROOT


def _lnk_paths() -> tuple[Path, Path, Path]:
    appdata = Path(os.environ.get("APPDATA", ""))
    programs = appdata / r"Microsoft\Windows\Start Menu\Programs"
    return (Path(os.path.expanduser("~")) / "Desktop" / f"{APP_NAME}.lnk",
            programs / f"{APP_NAME}.lnk",
            programs / "Startup" / f"{APP_NAME} Hotkey.lnk")


def _ps(value) -> str:
    """Escape a value for a single-quoted PowerShell string (paths may contain apostrophes)."""
    return str(value).replace("'", "''")


def _write_lnk(lnk: Path, target: str, args: str, description: str, icon: Path = None) -> bool:
    icon_line = f"$s.IconLocation = '{_ps(icon)},0';" if icon and icon.exists() else ""
    ps = (
        f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{_ps(lnk)}');"
        f"$s.TargetPath = '{_ps(target)}';"
        f"$s.Arguments = '{_ps(args)}';"
        f"$s.WorkingDirectory = '{_ps(ROOT)}';"
        f"$s.Description = '{_ps(description)}';"
        f"$s.WindowStyle = 7;"
        f"$s.Hotkey = '';"
        f"{icon_line}"
        f"$s.Save()"
    )
    res = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    return res.returncode == 0 and lnk.exists()


def _refresh_explorer_icons() -> None:
    """Tell Windows Explorer to reload shortcut icons (SHCNE_ASSOCCHANGED)."""
    try:
        import ctypes
        ctypes.windll.shell32.SHChangeNotify(0x08000000, 0x0000, None, None)
    except Exception:
        pass


def create_windows_shortcuts() -> list[Path]:
    """Desktop + Start Menu shortcuts to the widget, and a Startup shortcut for the hotkey launcher."""
    if sys.platform != "win32":
        return []
    pythonw = ROOT / ".venv" / "Scripts" / "pythonw.exe"
    py = str(pythonw) if pythonw.exists() else str(Path(sys.executable).with_name("pythonw.exe"))
    ico = ROOT / "assets" / "icon.ico"
    desktop, start_menu, startup = _lnk_paths()
    created = []
    for lnk, args, desc, icon in (
        (desktop, "-m eloquent --ui widget", f"{APP_NAME} - Live CPU Speech-to-Text", ico),
        (start_menu, "-m eloquent --ui widget", f"{APP_NAME} - Live CPU Speech-to-Text", ico),
        (startup, "-m eloquent --listen", f"{APP_NAME} launch hotkey", None),
    ):
        if lnk.parent.exists() and _write_lnk(lnk, py, args, desc, icon):
            created.append(lnk)
    if created:
        _refresh_explorer_icons()
    return created
