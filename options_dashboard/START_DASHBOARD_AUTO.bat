@echo off
title Options Greek Dashboard
echo.
echo  ========================================
echo   Options Greek Dashboard
echo  ========================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo  [!] Virtual environment not found. Run SETUP.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

:: Open browser after a short delay (gives the server time to start)
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8050"

echo  Starting dashboard...
echo  Browser will open automatically.
echo.
echo  Press Ctrl+C to stop.
echo.

python run.py

pause
