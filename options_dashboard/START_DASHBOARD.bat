@echo off
title Options Greek Dashboard
echo.
echo  ========================================
echo   Options Greek Dashboard
echo  ========================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo  [!] Virtual environment not found.
    echo      Run this first:  python -m venv venv
    echo      Then:            venv\Scripts\activate
    echo      Then:            pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo  Starting dashboard...
echo  Open your browser to:  http://localhost:8050
echo.
echo  Press Ctrl+C to stop.
echo.

python run.py

pause
