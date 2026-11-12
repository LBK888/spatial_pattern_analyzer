@echo off
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Run setup.bat first.
    exit /b 1
)
"venv\Scripts\python.exe" spatial_analysis_gui.py %*
exit /b %errorlevel%
