@echo off
title Options Dashboard - First Time Setup
echo.
echo  ========================================
echo   Options Dashboard - SETUP
echo  ========================================
echo.

cd /d "%~dp0"

echo  [1/3] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo.
    echo  [!] Failed to create venv. Make sure Python is installed.
    echo      Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo  [2/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo  [3/3] Installing dependencies (this may take a minute)...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo  [!] Some packages failed to install. Check the errors above.
    pause
    exit /b 1
)

echo.
echo  ========================================
echo   Setup complete!
echo  ========================================
echo.
echo   To start the dashboard:
echo     Double-click START_DASHBOARD.bat
echo.
echo   To test without IB:
echo     Double-click START_MOCK.bat
echo.
pause
