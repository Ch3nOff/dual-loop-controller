@echo off
setlocal enabledelayedexpansion
title Dual-Loop Autonomous Inference Server (vLLM OpenAI Compatible)

echo ===============================================================================
echo      DUAL-LOOP HIGH-THROUGHPUT vLLM INFERENCE SERVER (Car-Lift v4.5)
echo      PagedAttention, Continuous Batching ^& Dual-Loop Latent Deliberation
echo ===============================================================================
echo.

:: Detect Python executable
set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

echo [*] Using Python: %PYTHON_EXE%
"%PYTHON_EXE%" --version
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python or setup .venv.
    pause
    exit /b 1
)

echo.
echo -------------------------------------------------------------------------------
echo Select Model to Host:
echo   [1] Qwen/Qwen2.5-7B-Instruct (Recommended Production Baseline)
echo   [2] meta-llama/Llama-3.1-8B-Instruct (Meta LLaMA 3.1 8B)
echo   [3] Qwen/Qwen2.5-3B-Instruct (Ultra Fast Latent Deliberation)
echo   [4] Enter Custom Hugging Face Model ID
echo -------------------------------------------------------------------------------
set /p MODEL_CHOICE="Enter choice [1-4, Default=1]: "

if "%MODEL_CHOICE%"=="2" (
    set "SELECTED_MODEL=meta-llama/Llama-3.1-8B-Instruct"
) else if "%MODEL_CHOICE%"=="3" (
    set "SELECTED_MODEL=Qwen/Qwen2.5-3B-Instruct"
) else if "%MODEL_CHOICE%"=="4" (
    set /p SELECTED_MODEL="Enter Hugging Face Model ID: "
) else (
    set "SELECTED_MODEL=Qwen/Qwen2.5-7B-Instruct"
)

echo.
echo [Selected Model]: %SELECTED_MODEL%
echo.

set /p USER_PORT="Enter Server Port [Default=8000]: "
if "%USER_PORT%"=="" (
    set "USER_PORT=8000"
)

echo.
echo ===============================================================================
echo Starting Dual-Loop vLLM Inference Engine on http://127.0.0.1:%USER_PORT%/v1 ...
echo ===============================================================================
echo.
echo OpenAI Client / Hermes Agent Configuration:
echo   - Base URL: http://localhost:%USER_PORT%/v1
echo   - Model   : %SELECTED_MODEL%
echo   - API Key : not-needed (or any string on localhost)
echo.

"%PYTHON_EXE%" -m dual_loop.cli serve --model "%SELECTED_MODEL%" --port %USER_PORT%

pause
