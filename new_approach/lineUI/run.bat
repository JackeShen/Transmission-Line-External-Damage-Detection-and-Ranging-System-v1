@echo off
REM Launch the Power Line Distance Measurement UI
REM Uses the "goal" conda environment
cd /d "%~dp0..\.."
call conda run -n goal python -m new_approach.lineUI.app
pause
