@echo off
rem Eloquent - terminal mode
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m eloquent --ui terminal %*
) else (
    python -m eloquent --ui terminal %*
)
pause
