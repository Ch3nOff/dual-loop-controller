@echo off
title DUAL-LOOP HADL VS BASE MODEL - LIVE INFERENCE BENCHMARK
color 0B

echo ==============================================================================
echo       DUAL-LOOP HADL VS BASE MODEL: REAL-TIME INFERENCE BENCHMARK
echo               Active Inference Telemetry and Token Waste HUD
echo ==============================================================================
echo.

set "BENCH_DIR=C:\Users\Matthew Chen\Documents\bench"
set "PYTHON_EXE=C:\Users\Matthew Chen\Documents\X-Star\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found at: "%PYTHON_EXE%"
    echo Please verify that the virtual environment exists.
    pause
    exit /b 1
)

echo [OK] Python Environment : "%PYTHON_EXE%"
echo [OK] Working Directory  : "%BENCH_DIR%"
echo.
echo [*] Live Dashboard URL  : http://127.0.0.1:8000
echo [*] Launching web browser in 3 seconds...
echo [*] Loading Qwen3.5-2B model into RAM (~3.6s on CPU)...
echo [*] Press Ctrl+C in this terminal window to stop the server at any time.
echo ==============================================================================
echo.

start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:8000"

cd /d "%BENCH_DIR%"
"%PYTHON_EXE%" live_benchmark_server.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo ==============================================================================
    echo [ERROR] The server exited with error code %ERRORLEVEL%.
    echo ==============================================================================
) else (
    echo.
    echo ==============================================================================
    echo Benchmark server has stopped.
    echo ==============================================================================
)
pause
