@echo off
setlocal enabledelayedexpansion
title Dual-Loop Controller Installer & Setup

echo ===============================================================================
echo            DUAL-LOOP COGNITIVE CONTROLLER AUTOMATED INSTALLER
echo ===============================================================================
echo.

:: 1. Check Windows Long Paths Status
for /f "tokens=3" %%A in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled 2^>nul ^| findstr /i "LongPathsEnabled"') do (
    set "LONG_PATHS=%%A"
)

if not "%LONG_PATHS%"=="0x1" (
    echo [!] WARNING: Windows Long Path support is currently DISABLED.
    echo     Installing large packages like PyTorch directly into Windows site-packages
    echo     can trigger [Errno 2] MAX_PATH limit errors.
    echo.
    set /p FIX_LP="Would you like to enable Long Paths now (requires Administrator prompt)? [Y/N, Default=Y]: "
    if /i "!FIX_LP!"=="" set "FIX_LP=Y"
    if /i "!FIX_LP!"=="Y" (
        call fix_windows_longpaths.bat
    )
)

echo.
:: 2. Detect Python
set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

echo [*] Using Python: %PYTHON_EXE%
"%PYTHON_EXE%" --version
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+ from https://python.org.
    pause
    exit /b 1
)

:: 3. Detect NVIDIA GPU
set "HAS_CUDA=0"
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    set "HAS_CUDA=1"
    echo [*] NVIDIA GPU detected via nvidia-smi!
)

echo.
echo ===============================================================================
echo PyTorch Installation Mode:
if "!HAS_CUDA!"=="1" (
    echo   [1] PyTorch with NVIDIA CUDA 12.4 acceleration (Recommended for your GPU)
    echo   [2] PyTorch CPU only
    echo   [3] Skip PyTorch installation (already installed)
) else (
    echo   [1] PyTorch CPU only
    echo   [2] PyTorch with NVIDIA CUDA 12.4 acceleration
    echo   [3] Skip PyTorch installation (already installed)
)
echo ===============================================================================
set /p TORCH_CHOICE="Enter choice [1-3, Default=1]: "
if "%TORCH_CHOICE%"=="" set "TORCH_CHOICE=1"

if "%TORCH_CHOICE%"=="1" (
    if "!HAS_CUDA!"=="1" (
        echo [*] Installing PyTorch with CUDA 12.4...
        "%PYTHON_EXE%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    ) else (
        echo [*] Installing PyTorch CPU...
        "%PYTHON_EXE%" -m pip install torch
    )
) else if "%TORCH_CHOICE%"=="2" (
    if "!HAS_CUDA!"=="1" (
        echo [*] Installing PyTorch CPU...
        "%PYTHON_EXE%" -m pip install torch
    ) else (
        echo [*] Installing PyTorch with CUDA 12.4...
        "%PYTHON_EXE%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    )
)

echo.
echo [*] Installing Dual-Loop Controller with server & CLI capabilities...
"%PYTHON_EXE%" -m pip install -e .

echo.
echo ===============================================================================
echo [SUCCESS] Dual-Loop Controller is installed and ready!
echo Run the server anytime with: START_SERVER.bat or 'hadl serve'
echo ===============================================================================
echo.
pause
