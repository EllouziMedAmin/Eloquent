@echo off
rem Eloquent one-time setup: virtual environment, dependencies, shortcuts and the launch hotkey.
cd /d "%~dp0"

where python >nul 2>nul || (
    echo Python 3.10+ is required: https://www.python.org/downloads/
    echo During install, tick "Add python.exe to PATH", then run setup.bat again.
    pause & exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv || (pause & exit /b 1)
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt || (pause & exit /b 1)

if not exist ".env" if exist ".env.example" copy ".env.example" ".env" >nul

rem Desktop + Start Menu shortcuts, Startup entry, and start the hotkey launcher now.
".venv\Scripts\python.exe" -m eloquent --create-shortcut

echo.
echo ==================================================================
echo  Eloquent is ready.
echo    Press Ctrl+Shift+E anywhere to start it (works after reboot too)
echo    or double-click the "Eloquent" icon on your Desktop.
echo    The speech model (~670 MB) downloads on the first launch.
echo    Optional: add a free Gemini key in the widget: ... menu - Settings
echo ==================================================================
pause
