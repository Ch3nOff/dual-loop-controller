@echo off
setlocal enabledelayedexpansion
title HADL v2.4.0 - Large-Scale Empirical & Use-Case Benchmark Suite
color 0A

echo ==============================================================================
echo   HADL v2.4.0: LARGE-SCALE EMPIRICAL & USE-CASE BENCHMARK SUITE
echo   Autopoietic Dual-Loop Cognitive Architecture
echo   Real PyTorch Computations (N = 2,500 Evaluations)
echo ==============================================================================
echo.

cd /d "%~dp0"

:: 1. Detect Python virtual environment or system Python
if exist ".\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=.\.venv\Scripts\python.exe"
    echo [*] Using local virtual environment: .\.venv\Scripts\python.exe
) else (
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        set "PYTHON_EXE=python"
        echo [*] Using system python: python
    ) else (
        where py >nul 2>nul
        if %errorlevel% equ 0 (
            set "PYTHON_EXE=py"
            echo [*] Using Python launcher: py
        ) else (
            color 0C
            echo [!] ERROR: Python could not be found. Please ensure Python is installed and in PATH.
            pause
            exit /b 1
        )
    )
)

echo [*] Starting Large-Scale Benchmark Suite...
echo.

"%PYTHON_EXE%" dual_loop/benchmarks/large_scale_usecase_benchmark.py

if %errorlevel% neq 0 (
    color 0C
    echo.
    echo [!] Benchmark encountered an error during execution.
    pause
    exit /b %errorlevel%
)

echo.
echo ==============================================================================
echo   BENCHMARK RUN COMPLETED SUCCESSFULLY!
echo ==============================================================================
echo [*] Results saved to:
echo     - eval_results\large_scale_usecase_benchmark.json
echo     - bench\large_scale_usecase_benchmark_results.json
echo     - large_scale_usecase_benchmark_graph.png
echo.

set /p OPEN_GRAPH="Do you want to open the generated 6-panel visualization graph now? [Y/n]: "
if /i "%OPEN_GRAPH%"=="n" goto :DONE
if exist "large_scale_usecase_benchmark_graph.png" (
    start "" "large_scale_usecase_benchmark_graph.png"
)

:DONE
echo.
echo Press any key to exit...
pause >nul
