@echo off
setlocal
cd /d "%~dp0\.."

echo ============================================================
echo   Starting GPS Signal Simulator Server on Windows...
echo ============================================================

if "%ALLOW_TX%"=="" set ALLOW_TX=1
if "%RF_FRONTEND_ENABLED%"=="" set RF_FRONTEND_ENABLED=1

echo Environment: ALLOW_TX=%ALLOW_TX% RF_FRONTEND_ENABLED=%RF_FRONTEND_ENABLED%

if exist ".venv\Scripts\activate.bat" goto use_venv
goto use_system

:use_venv
echo Activating virtual environment...
call ".venv\Scripts\activate.bat"
goto start_server

:use_system
echo Using system Python environment...
goto start_server

:start_server
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH!
    echo Please install Python 3.10+ and ensure python is in PATH.
    pause
    exit /b 1
)

echo.
echo Starting server at http://127.0.0.1:8000
echo Open your browser and navigate to: http://127.0.0.1:8000
echo Press Ctrl+C in this window to stop the server.
echo.

python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Server exited with code %ERRORLEVEL%.
    pause
)
