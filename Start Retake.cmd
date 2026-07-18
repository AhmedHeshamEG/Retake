@echo off
title RETAKE
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Local .venv was not found.
    echo I expected: .venv\Scripts\python.exe
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" retake.py

echo.
echo RETAKE stopped.
pause
