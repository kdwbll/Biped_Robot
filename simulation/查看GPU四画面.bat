@echo off
cd /d "%~dp0"
echo Live GPU training worker snapshots, with virtual assistance.
.venv\Scripts\python.exe view_parallel_progress.py --gpu
pause
