@echo off
title HADL Cognitive Runtime Cockpit v2.4.0
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:MENU
cls
echo ===============================================================================
echo            HADL COGNITIVE RUNTIME COCKPIT ^| AUTOPOIETIC DUAL-LOOP v2.4.0
echo            Hardware-Aligned Latent Deliberation ^& Autonomous Reasoning
echo ===============================================================================
echo.

rem 1. Check Python virtual environment
if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    set "PYTHON_EXEC=%SCRIPT_DIR%.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=python"
)

echo [*] Working Directory: %SCRIPT_DIR%
echo [*] Python Interpreter: !PYTHON_EXEC!
echo.
echo Select an option:
echo   [1] Launch Cockpit with Auto-Detected Backbone (GLM-4 / Qwen + Adapter)
echo   [2] Launch Cockpit in Instant Fast Mode (Zero-Download Engine)
echo   [3] Package ^& Publish GLM-4 Adapter to Hugging Face Hub
echo   [4] Package ^& Publish Qwen Adapter to Hugging Face Hub
echo   [5] Exit
echo.
set /p OPTION="Enter choice [1-5, default=1]: "
if "!OPTION!"=="" set "OPTION=1"

if "!OPTION!"=="1" (
    echo.
    echo ===============================================================================
    echo [*] Starting HADL Runtime Server (Auto-Detecting Backbone ^& Adapter)...
    echo [*] UI Dashboard will open at: http://127.0.0.1:8000/
    echo [*] Press Ctrl+C in this window anytime to stop the server.
    echo ===============================================================================
    start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:8000"
    "!PYTHON_EXEC!" -m dual_loop.cli serve --host 127.0.0.1 --port 8000
    echo.
    echo [*] Server process terminated.
    echo.
    pause
    goto MENU
)

if "!OPTION!"=="2" (
    echo.
    echo ===============================================================================
    echo [*] Starting HADL Instant Fast Cockpit Server...
    echo [*] UI Dashboard will open at: http://127.0.0.1:8000/
    echo [*] Press Ctrl+C in this window anytime to stop the server.
    echo ===============================================================================
    start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"
    "!PYTHON_EXEC!" -m dual_loop.cli serve --mock --host 127.0.0.1 --port 8000
    echo.
    echo [*] Server process terminated.
    echo.
    pause
    goto MENU
)

if "!OPTION!"=="3" (
    echo.
    echo [*] Packaging GLM-4 Adapter Bundle for Hugging Face Hub...
    "!PYTHON_EXEC!" -m dual_loop.cli publish-hf --model-type glm4
    echo.
    pause
    goto MENU
)

if "!OPTION!"=="4" (
    echo.
    echo [*] Packaging Qwen Adapter Bundle for Hugging Face Hub...
    "!PYTHON_EXEC!" -m dual_loop.cli publish-hf --model-type qwen
    echo.
    pause
    goto MENU
)

if "!OPTION!"=="5" (
    echo Goodbye!
    exit /b 0
)

echo [!] Invalid selection. Please choose 1, 2, 3, 4, or 5.
pause
goto MENU
