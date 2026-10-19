@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Ensure Python 3.10+ is installed and on PATH.
        pause
        exit /b 1
    )
    echo [INFO] Upgrading pip...
    "%PYTHON%" -m pip install --upgrade pip
    echo [INFO] Installing requirements...
    "%PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install requirements.
        pause
        exit /b 1
    )
)

echo [INFO] Checking AI models...
"%PYTHON%" backend\download_models.py
if errorlevel 1 (
    echo [ERROR] Model download or validation failed.
    pause
    exit /b 1
)

echo [INFO] Starting application at http://127.0.0.1:8000 ...
cd backend
"%PYTHON%" -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
