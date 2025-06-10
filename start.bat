@echo off
echo Starting Spatial Analysis Tool...

:: 檢查虛擬環境是否存在
if not exist venv (
    echo ERROR：virtual env python not found。
    echo please run setup.bat first。
    pause
    exit /b 1
)

:: 啟動虛擬環境
call venv\Scripts\activate.bat

:: 檢查必要的檔案是否存在
if not exist spatial_analysis_gui.py (
    echo ERROR：Missing spatial_analysis_gui.py
    pause
    exit /b 1
)

:: 執行主程式
echo Starting Main GUI...
python spatial_analysis_gui.py

:: 如果程式異常結束，顯示錯誤訊息
if errorlevel 1 (
    echo.
    echo 程式異常結束，請檢查錯誤訊息。
    pause
) 