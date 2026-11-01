@echo off
cd /d "%~dp0"
echo SIMULATION: virtual support carries 64 percent of robot weight.
echo A virtual balance torque is also enabled. Not autonomous walking.
.venv\Scripts\python.exe view_64pct_progress.py
pause
