@echo off
cd /d "%~dp0"
echo Simulation demo uses virtual support for 80 percent of robot weight.
echo This is not unsupported walking and must not be used on hardware.
.venv\Scripts\python.exe view_assisted_imitation.py
pause
