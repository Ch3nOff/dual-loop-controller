@echo off
setlocal enabledelayedexpansion

echo ===============================================================================
echo   HADL v3.0: Web Game Creation, Continual Learning, and Real Output Benchmark
echo ===============================================================================
echo.

:: Detect Python executable
set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

echo [*] Using Python: %PYTHON_EXE%
%PYTHON_EXE% --version
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python or setup .venv.
    pause
    exit /b 1
)

echo.
echo [*] Executing unconstrained web game generation benchmark...
echo [*] Comparing:
echo     1. Naive Base Model (No Dual-Loop)
echo     2. HADL Dual-Loop (Before Learning / Zero-Shot)
echo     3. In-situ Teaching Phase (Plastic Synaptic Fast-Weights + Sleep Consolidation)
echo     4. HADL Dual-Loop Post-Teaching ('Suruh Coba Ulang')
echo.

"%PYTHON_EXE%" scripts\run_webgame_comparison.py
if errorlevel 1 (
    echo.
    echo [FAILED] Benchmark execution encountered an error.
    pause
    exit /b 1
)

echo.
echo ===============================================================================
echo   BENCHMARK COMPLETED SUCCESSFULLY!
echo ===============================================================================
echo.
echo Real Playable HTML Games Generated in:
echo   - eval_results\games\game_1_naive_base.html
echo   - eval_results\games\game_2_hadl_before_learning.html
echo   - eval_results\games\game_3_hadl_after_learning.html
echo   - eval_results\games\arena_viewer.html (Side-by-Side Arena Portal)
echo.
echo Telemetry and Reports:
echo   - eval_results\webgame_comparison_benchmark.json
echo   - eval_results\webgame_comparison_report.png
echo.
echo [*] Launching Interactive Arena Viewer in your default web browser...
start "" "eval_results\games\arena_viewer.html"

echo.
echo Done. Press any key to exit.
pause >nul
