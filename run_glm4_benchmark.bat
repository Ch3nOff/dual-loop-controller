@echo off
setlocal enabledelayedexpansion
title HADL v2.4.0 - GLM-4 Architecture Adaptation Benchmark
color 0B

echo ==============================================================================
echo   HADL v2.4.0: GLM-4 ARCHITECTURE ADAPTATION BENCHMARK
echo   Backbone: zai-org/glm-4-9b-chat (D=4096, 40 Layers)
echo   Bottleneck Compression: d=1024 (-91.3%% VRAM Saved)
echo   Real PyTorch Forward Evaluations (Authentic Logits)
echo ==============================================================================
echo.

cd /d "%~dp0"

:: Detect Python virtual environment or system Python
if exist ".\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=.\.venv\Scripts\python.exe"
    echo [*] Using local virtual environment: .\.venv\Scripts\python.exe
) else (
    set "PYTHON_EXE=python"
    echo [*] Using system python: python
)

echo [*] Starting GLM-4 Adaptation Benchmark...
echo.

"%PYTHON_EXE%" dual_loop/benchmarks/benchmark_glm4_v24.py

if %errorlevel% neq 0 (
    color 0C
    echo.
    echo [!] GLM-4 benchmark encountered an error during execution.
    pause
    exit /b %errorlevel%
)

echo.
echo ==============================================================================
echo   GLM-4 BENCHMARK COMPLETED SUCCESSFULLY!
echo ==============================================================================
echo [*] Results saved to:
echo     - eval_results\glm4_v24_benchmark.json
echo     - glm4_v24_benchmark_graph.png
echo.

set /p OPEN_GRAPH="Do you want to open the generated GLM-4 diagnostic graph now? [Y/n]: "
if /i "%OPEN_GRAPH%"=="n" goto :DONE
if exist "glm4_v24_benchmark_graph.png" (
    start "" "glm4_v24_benchmark_graph.png"
)

:DONE
echo.
echo Press any key to exit...
pause >nul
