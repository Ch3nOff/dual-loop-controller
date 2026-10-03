@echo off
setlocal enabledelayedexpansion
title Dual-Loop Autonomous Inference Server (OpenAI Compatible)

echo ===============================================================================
echo      DUAL-LOOP COGNITIVE OS INFERENCE SERVER (vLLM / Ollama Alternative)
echo      Hardware-Aligned Dynamic VRAM Auto-Tuning ^& Zero OOM Guarantee
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
echo [*] Analyzing Hardware Profile and Available VRAM...
"%PYTHON_EXE%" -c "from dual_loop.server.vram_tuner import VRAMAutoTuner; hw = VRAMAutoTuner.profile_hardware(); print(f'    GPU Detected   : {hw.device_name}\n    Total VRAM     : {hw.total_vram_gib} GiB\n    Free VRAM      : {hw.free_vram_gib} GiB\n    Host RAM Total : {hw.host_ram_total_gib} GiB')"
echo.

echo -------------------------------------------------------------------------------
echo Select Model to Host:
echo   [1] Qwen/Qwen3.5-2B (Instant Local Cache - 1.91 GiB VRAM on RTX 5060!)
echo   [2] meta-llama/Llama-3.1-8B-Instruct (4-bit NF4 / BF16 Auto-Adapted)
echo   [3] Qwen/Qwen2.5-7B-Instruct (4-bit NF4 / BF16 Auto-Adapted)
echo   [4] Qwen/Qwen2.5-3B-Instruct (Ultra Fast Latent Deliberation)
echo   [5] Enter Custom Hugging Face Model ID
echo -------------------------------------------------------------------------------
set /p MODEL_CHOICE="Enter choice [1-5, Default=1]: "

if "%MODEL_CHOICE%"=="2" (
    set "SELECTED_MODEL=meta-llama/Llama-3.1-8B-Instruct"
) else if "%MODEL_CHOICE%"=="3" (
    set "SELECTED_MODEL=Qwen/Qwen2.5-7B-Instruct"
) else if "%MODEL_CHOICE%"=="4" (
    set "SELECTED_MODEL=Qwen/Qwen2.5-3B-Instruct"
) else if "%MODEL_CHOICE%"=="5" (
    set /p SELECTED_MODEL="Enter Hugging Face Model ID (e.g. meta-llama/Llama-3.1-8B-Instruct): "
) else (
    set "SELECTED_MODEL=Qwen/Qwen3.5-2B"
)

echo.
echo [Selected Model]: %SELECTED_MODEL%
echo.
set /p USER_HEADROOM="Enter VRAM Safety Headroom in GiB (e.g. 4.0 for 12GB GPU, or 'auto') [Default=auto]: "
if "%USER_HEADROOM%"=="" (
    set "USER_HEADROOM=auto"
)

set /p USER_PORT="Enter Server Port [Default=8000]: "
if "%USER_PORT%"=="" (
    set "USER_PORT=8000"
)

echo.
echo ===============================================================================
echo Starting Dual-Loop Inference Engine on http://127.0.0.1:%USER_PORT%/v1 ...
echo ===============================================================================
echo.
echo Hermes Agent / OpenAI Client Configuration:
echo   - Base URL: http://localhost:%USER_PORT%/v1
echo   - Model   : %SELECTED_MODEL%
echo   - API Key : not-needed (or any string)
echo.

"%PYTHON_EXE%" -m dual_loop.cli serve --model "%SELECTED_MODEL%" --port %USER_PORT% --headroom "%USER_HEADROOM%"

pause
