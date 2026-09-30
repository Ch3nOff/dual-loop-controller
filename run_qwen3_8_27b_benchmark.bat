@echo off
setlocal enabledelayedexpansion

echo ===============================================================================
echo   Qwen/Qwen3.8-27B Dedicated Hardware Memory & OOM Comparative Benchmark
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
echo [*] Executing Qwen/Qwen3.8-27B benchmark across 4 versions/regimes:
echo     1. Qwen3.8-27B Native BF16 (Pure GPU - Monitoring OOM crash)
echo     2. Qwen3.8-27B Q4 Quantized (Pure GPU - Monitoring OOM crash)
echo     3. Qwen3.8-27B Q4 with CPU Offload & Host RAM Swapping
echo     4. Qwen3.8-27B + HADL Dual-Loop Holographic Latent Compression
echo.

"%PYTHON_EXE%" scripts\run_qwen3_8_27b_benchmark.py
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
echo Real Playable Games Generated:
echo   - eval_results\games\game_qwen3_8_27b_cpu_offload.html
echo   - eval_results\games\game_qwen3_8_27b_hadl_hologram.html
echo   - eval_results\games\qwen3_8_arena_viewer.html (OOM and Playable Arena)
echo.
echo Telemetry and Reports:
echo   - eval_results\qwen3_8_27b_oom_benchmark.json
echo   - eval_results\qwen3_8_27b_oom_comparison.png
echo.
echo [*] Launching Interactive Qwen3.8-27B Arena Viewer in default browser...
start "" "eval_results\games\qwen3_8_arena_viewer.html"

echo.
echo Done. Press any key to exit.
pause >nul
