@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0view_smooth_hip_standing.py"
if errorlevel 1 pause
