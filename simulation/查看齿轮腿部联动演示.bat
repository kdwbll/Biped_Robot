@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0view_gear_kinematic_preview.py"
if errorlevel 1 pause
