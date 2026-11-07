@echo off
rem Repertory: starts the service in the background if it is not running, and opens the dashboard.
rem Never run a same-named program from this folder when a command is looked up by name.
set NoDefaultCurrentDirectoryInExePath=1
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Creating the Windows-side Python environment...
  python -m venv .venv || (echo Python 3 is required: https://www.python.org/downloads/ & pause & exit /b 1)
)
".venv\Scripts\python.exe" -c "import psutil, win32pdh, win32job" 2>nul || ".venv\Scripts\python.exe" -m pip install -q --require-hashes -r repertory\requirements.txt
start "" ".venv\Scripts\pythonw.exe" -m repertory open
