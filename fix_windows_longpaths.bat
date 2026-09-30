@echo off
:: Batch script to enable Windows Long Path support (MAX_PATH limit removal)
:: Requires Administrator privileges.

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Requesting Administrator privileges to enable Windows Long Paths...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process cmd -ArgumentList '/c \"\"%~f0\"\"' -Verb RunAs"
    exit /b
)

echo ===============================================================================
echo            ENABLING WINDOWS LONG PATH SUPPORT (LongPathsEnabled)
echo ===============================================================================
echo.
echo [*] Applying registry fix to remove the 260-character MAX_PATH limit...

reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" /t REG_DWORD /d 1 /f

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Windows Long Paths enabled successfully!
    echo [*] You can now install PyTorch and Dual-Loop Controller without [Errno 2] path errors.
) else (
    echo.
    echo [ERROR] Failed to modify registry. Please run this script as Administrator.
)

echo.
pause
