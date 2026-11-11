@echo off
rem One-time setup: creates .venv and installs dependencies. Models download on first launch.
cd /d "%~dp0"
where python >nul 2>nul || (echo Python 3.10+ is required: https://www.python.org/downloads/ & pause & exit /b 1)
python -m venv .venv || (pause & exit /b 1)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt || (pause & exit /b 1)
echo.
echo Setup done. Start Eloquent with Eloquent.bat (widget) or Eloquent-Terminal.bat.
pause
