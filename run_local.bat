@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PY_EXE=.venv\Scripts\python.exe"
) else (
    set "PY_EXE=python"
)

echo Using Python: %PY_EXE%
echo Starting GTA VI AI AutoClipper on http://127.0.0.1:17430 ...
start "" http://127.0.0.1:17430
"%PY_EXE%" run.py
pause
