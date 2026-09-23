@echo off
title HADL Cognitive Runtime Cockpit v2.4.0
setlocal

cd /d "%~dp0"

:MENU
cls
echo ===============================================================================
echo            HADL COGNITIVE RUNTIME COCKPIT - AUTOPOIETIC DUAL-LOOP v2.4.0
echo            Hardware-Aligned Latent Deliberation and Autonomous Reasoning
echo ===============================================================================
echo.

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXEC=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=python"
)

echo [*] Working Directory : %~dp0
echo [*] Python Interpreter: %PYTHON_EXEC%
echo.
echo Select an option:
echo   [1] Launch Cockpit with Auto-Detected Backbone and Adapter
echo   [2] Launch Cockpit in Instant Fast Mode (Zero-Download Engine)
echo   [3] Package and Publish GLM-4 Adapter to Hugging Face Hub
echo   [4] Package and Publish Qwen Adapter to Hugging Face Hub
echo   [5] Exit
echo.
set "OPTION=1"
set /p "OPTION=Enter choice [1-5, default=1]: "

if "%OPTION%"=="1" goto RUN_AUTO
if "%OPTION%"=="2" goto RUN_MOCK
if "%OPTION%"=="3" goto PUB_GLM
if "%OPTION%"=="4" goto PUB_QWEN
if "%OPTION%"=="5" goto QUIT

echo.
echo [!] Invalid selection: %OPTION%. Please choose 1, 2, 3, 4, or 5.
pause
goto MENU

:RUN_AUTO
echo.
echo ===============================================================================
echo [*] Initializing HADL Runtime Server...
echo [*] Auto-detecting model architecture and loading adapter weights...
echo [*] UI Dashboard will open at: http://127.0.0.1:8000/
echo [*] Press Ctrl+C in this window anytime to stop the server.
echo ===============================================================================
start "" cmd /c "timeout /t 4 /nobreak >nul & start http://127.0.0.1:8000"
"%PYTHON_EXEC%" -m dual_loop.cli serve --host 127.0.0.1 --port 8000
echo.
echo ===============================================================================
echo [*] Server process stopped.
echo ===============================================================================
pause
goto MENU

:RUN_MOCK
echo.
echo ===============================================================================
echo [*] Starting HADL Instant Fast Cockpit Server (Zero Wait)...
echo [*] UI Dashboard will open at: http://127.0.0.1:8000/
echo [*] Press Ctrl+C in this window anytime to stop the server.
echo ===============================================================================
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"
"%PYTHON_EXEC%" -m dual_loop.cli serve --mock --host 127.0.0.1 --port 8000
echo.
echo ===============================================================================
echo [*] Server process stopped.
echo ===============================================================================
pause
goto MENU

:PUB_GLM
echo.
echo ===============================================================================
echo [*] Packaging GLM-4 Adapter Bundle for Hugging Face Hub...
echo ===============================================================================
"%PYTHON_EXEC%" -m dual_loop.cli publish-hf --model-type glm4
echo.
pause
goto MENU

:PUB_QWEN
echo.
echo ===============================================================================
echo [*] Packaging Qwen Adapter Bundle for Hugging Face Hub...
echo ===============================================================================
"%PYTHON_EXEC%" -m dual_loop.cli publish-hf --model-type qwen
echo.
pause
goto MENU

:QUIT
echo.
echo Goodbye!
exit /b 0
