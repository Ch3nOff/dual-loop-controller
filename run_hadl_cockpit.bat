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
echo   [1] Launch with Qwen-3.5-2B (Complete Weights in Cache + Trained Adapter) [RECOMMENDED]
echo   [2] Launch with GLM-4-9B (d=1024 Bottleneck Adapter)
echo   [3] Launch in Instant Fast Mode (Zero Wait Demo Engine)
echo   [4] Package and Publish GLM-4 Adapter to Hugging Face Hub
echo   [5] Package and Publish Qwen Adapter to Hugging Face Hub
echo   [6] Exit
echo.
set "OPTION=1"
set /p "OPTION=Enter choice [1-6, default=1]: "

if "%OPTION%"=="1" goto RUN_QWEN
if "%OPTION%"=="2" goto RUN_GLM4
if "%OPTION%"=="3" goto RUN_MOCK
if "%OPTION%"=="4" goto PUB_GLM
if "%OPTION%"=="5" goto PUB_QWEN
if "%OPTION%"=="6" goto QUIT

echo.
echo [!] Invalid selection: %OPTION%. Please choose 1, 2, 3, 4, 5, or 6.
pause
goto MENU

:RUN_QWEN
echo.
echo ===============================================================================
echo [*] Initializing HADL Runtime Server with Qwen-3.5-2B + Deliberation Adapter...
echo [*] UI Dashboard will open at: http://127.0.0.1:8000/
echo [*] Press Ctrl+C in this window anytime to stop the server.
echo ===============================================================================
start "" cmd /c "timeout /t 4 /nobreak >nul & start http://127.0.0.1:8000"
"%PYTHON_EXEC%" -m dual_loop.cli serve --model Qwen/Qwen3.5-2B --host 127.0.0.1 --port 8000
echo.
echo ===============================================================================
echo [*] Server process stopped.
echo ===============================================================================
pause
goto MENU

:RUN_GLM4
echo.
echo ===============================================================================
echo [*] Initializing HADL Runtime Server with GLM-4-9B Bottleneck Adapter...
echo [*] UI Dashboard will open at: http://127.0.0.1:8000/
echo [*] Press Ctrl+C in this window anytime to stop the server.
echo ===============================================================================
start "" cmd /c "timeout /t 4 /nobreak >nul & start http://127.0.0.1:8000"
"%PYTHON_EXEC%" -m dual_loop.cli serve --model zai-org/glm-4-9b-chat --bottleneck-dim 1024 --host 127.0.0.1 --port 8000
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
