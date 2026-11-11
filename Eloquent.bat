@echo off
rem Eloquent - floating widget (no console window)
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" -m eloquent --ui widget %*
) else (
    start "" pythonw -m eloquent --ui widget %*
)
