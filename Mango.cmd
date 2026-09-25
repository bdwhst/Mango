@echo off
rem Mango GUI (docs/DESIGN.md 6.7): double-click to play a model or analyse a game.
rem Needs the venv (scripts\setup_env.ps1) and the CUDA build (scripts\build.ps1 -Torch).
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo The Python venv is missing: run scripts\setup_env.ps1 first.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "scripts\gui.py" %*
