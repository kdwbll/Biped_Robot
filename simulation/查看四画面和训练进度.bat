@echo off
cd /d "%~dp0"
echo Live snapshots from four training workers. Virtual support is enabled.
.venv\Scripts\python.exe view_parallel_progress.py
pause
