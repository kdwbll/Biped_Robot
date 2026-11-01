@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0view_ppo.py" ppo_experiment_4096.zip
if errorlevel 1 pause
