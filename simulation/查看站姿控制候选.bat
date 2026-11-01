@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0view_standing_candidate.py"
if errorlevel 1 pause
