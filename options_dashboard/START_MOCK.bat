@echo off
title Options Dashboard (MOCK MODE)
echo.
echo  ========================================
echo   Options Greek Dashboard - MOCK MODE
echo   (No IB connection needed)
echo  ========================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo  [!] Virtual environment not found.
    echo      Run SETUP.bat first.
    echo.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo  Starting dashboard with synthetic data...
echo  Open your browser to:  http://localhost:8050
echo.
echo  Press Ctrl+C to stop.
echo.

python run.py --mock

pause
