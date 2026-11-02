@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0view_foot_suppressed_plate.py"
if errorlevel 1 pause
