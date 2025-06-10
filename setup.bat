@echo off
chcp 65001
echo 正在設置空間模式分析工具環境...

:: 檢查 Python 是否已安裝
python --version >nul 2>&1
if errorlevel 1 (
    echo 錯誤：未找到 Python。請先安裝 Python 3.10 或更新版本。
    pause
    exit /b 1
)

:: 創建虛擬環境
echo 正在創建 Python 虛擬環境...
python -m venv venv
if errorlevel 1 (
    echo 錯誤：創建虛擬環境失敗。
    pause
    exit /b 1
)

:: 啟動虛擬環境
echo 正在啟動虛擬環境...
call venv\Scripts\activate.bat

:: 升級 pip
echo 正在升級 pip...
python -m pip install --upgrade pip

:: 安裝必要的套件
echo 正在安裝必要的套件...
pip install pandas numpy matplotlib scipy openpyxl tqdm pillow scikit-learn

:: 檢查安裝是否成功
echo 正在檢查安裝...
python -c "import pandas, numpy, matplotlib, scipy, openpyxl, tqdm, PIL, sklearn" >nul 2>&1
if errorlevel 1 (
    echo 警告：某些套件可能未正確安裝。
) else (
    echo 所有套件已成功安裝。
)

echo.
echo Installation Complete!
echo Please use start.bat to start。
echo.
pause 