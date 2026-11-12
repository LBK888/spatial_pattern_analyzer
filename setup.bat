@echo off
setlocal
cd /d "%~dp0"
python -c "import sys; sys.exit(sys.version_info < (3,10))" || goto fail
if not exist "venv\Scripts\python.exe" python -m venv venv || goto fail
"venv\Scripts\python.exe" -m pip install -r requirements.txt || goto fail
echo Setup complete. Run start.bat.
exit /b 0
:fail
echo Setup failed. Check that Python 3.10 or newer is installed.
exit /b 1
