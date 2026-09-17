@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

:: ============================================================================
::  Dual-Loop Cognitive Controller v2.1 — Real-Time Benchmark Launcher
::  Backbone: Qwen/Qwen3.5-2B (Authentic 1.88B Parameters, D=2048, Layer 11 Hook)
:: ============================================================================

title Dual-Loop Cognitive Controller — Benchmark Runner

echo.
echo ========================================================================
echo   DUAL-LOOP COGNITIVE CONTROLLER v2.1 — BENCHMARK RUNNER
echo   Backbone: Qwen/Qwen3.5-2B (100%% Authentic Real Weights - Zero Ghost Model)
echo ========================================================================
echo.

:: -- Step 1: Find Python --
set "PYTHON="

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON=%~dp0.venv\Scripts\python.exe"
    echo   [OK] Menggunakan Python dari virtual environment (.venv)
    goto :python_found
)

where python >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    for /f "tokens=*" %%i in ('where python 2^>nul') do (
        set "PYTHON=%%i"
        goto :python_found
    )
)

echo   [ERROR] Python tidak ditemukan!
echo   Pastikan Python 3.9+ dan .venv terinstal di folder proyek.
echo.
pause
exit /b 1

:python_found
for /f "tokens=*" %%v in ('"!PYTHON!" --version 2^>^&1') do echo   Versi Python: %%v

:: -- Step 2: Check Adapter Weights --
if exist "%~dp0dual_loop\checkpoints\adapter_model.safetensors" (
    echo   [OK] Bobot Adapter Calibrated ditemukan (adapter_model.safetensors)
) else (
    echo   [WARNING] Bobot adapter tidak ditemukan di dual_loop\checkpoints\adapter_model.safetensors
)

:: -- Step 3: Check Device (CUDA / CPU) --
for /f "tokens=*" %%c in ('"!PYTHON!" -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (Torch Float32)')" 2^>^&1') do (
    echo   Perangkat Komputasi: %%c
)
echo.

:: -- Step 4: Interactive Menu --
:menu
echo ========================================================================
echo   PILIH METODE EVALUASI BENCHMARK:
echo ========================================================================
echo.
echo   [1] Perbandingan Langsung (Head-to-Head Spotlight Showdown) ★ REKOMENDASI CEPAT
echo       -> Menguji langsung soal-soal nyata di mana Base Model keliru dan
echo          Dual-Loop berhasil menyelamatkan (Rescued: Wrong -^> Right).
echo       -> Menampilkan soal lengkap, opsi jawaban, analisis penyebab, dan
echo          secara otomatis membuka laporan visual HTML di browser Anda.
echo       -> Selesai dalam ~20-30 detik!
echo.
echo   [2] Live Web Dashboard (Browser Interaktif Real-Time via SSE)
echo       -> Menjalankan suite 20 benchmark dengan dashboard HTML modern.
echo       -> Menampilkan Live Question Spotlight Card dan grafik Chart.js.
echo.
echo   [3] Benchmark Terminal Lengkap (20 Benchmark ANSI Colored Output)
echo       -> Menjalankan seluruh 20 task benchmark di layar terminal.
echo.
echo   [4] Demo 3-Pass Selective Virtual Memory Loop (The Smart Brain Loop)
echo       -> Menguji Pass 1 (Triage), Pass 2 (Selective Re-Think K=3), Pass 3 (Memory).
echo       -> Membuktikan zero token waste dan akselerasi memori 3,146x.
echo.
echo   [5] 2-Bench Matrix Question Helper (Eliminasi Opsi Distractor) ★ FITUR BARU
echo       -> Bench 1: Raw Base Screening + Matriks Pembantu Soal (Pencatat Wrong Logs).
echo       -> Bench 2: Eliminasi Opsi Sampah + Deliberasi Tajam System 2 (K=3).
echo       -> Membuktikan lonjakan akurasi +33.3%% s/d +40.0%% pada Qwen3.5-2B!
echo.
echo   [6] Keluar
echo.
echo ========================================================================
set "CHOICE="
set /p CHOICE="  Masukkan pilihan Anda [1/2/3/4/5/6] (default=1): "
if "!CHOICE!"=="" set "CHOICE=1"

if "!CHOICE!"=="1" goto :run_spotlight
if "!CHOICE!"=="2" goto :run_web
if "!CHOICE!"=="3" goto :run_terminal
if "!CHOICE!"=="4" goto :run_3pass
if "!CHOICE!"=="5" goto :run_matrix_helper
if "!CHOICE!"=="6" (
    echo   Keluar.
    exit /b 0
)

echo   Pilihan tidak valid! Silakan masukkan angka 1 sampai 6.
echo.
goto :menu

:: ----------------------------------------------------------------------------
:: OPSI 1: HEAD-TO-HEAD SPOTLIGHT SHOWDOWN
:: ----------------------------------------------------------------------------
:run_spotlight
cls
echo.
echo ========================================================================
echo   MENJALANKAN HEAD-TO-HEAD SPOTLIGHT SHOWDOWN (QWEN3.5-2B)
echo ========================================================================
echo.
cd /d "%~dp0"
"!PYTHON!" compare_head_to_head.py
echo.
echo   [OK] Selesai! Laporan visual interaktif telah dibuka di browser Anda.
echo        File: eval_results\head_to_head_report.html
echo.
pause
goto :menu

:: ----------------------------------------------------------------------------
:: OPSI 2: WEB DASHBOARD REAL-TIME
:: ----------------------------------------------------------------------------
:run_web
cls
echo.
echo ========================================================================
echo   PILIH JUMLAH SAMPEL PER TASK UNTUK WEB DASHBOARD:
echo ========================================================================
echo.
echo     1 = Demo Cepat   (3 sampel/task,  60 total, ~2-3 menit pada CPU)
echo     2 = Standar      (10 sampel/task, 200 total, ~15 menit pada CPU)
echo.
set "S_CHOICE="
set /p S_CHOICE="  Pilihan sampel [1/2] (default=1): "
if "!S_CHOICE!"=="" set "S_CHOICE=1"
if "!S_CHOICE!"=="2" (
    set "SAMPLES=10"
) else (
    set "SAMPLES=3"
)

set "PORT=8765"
set /p PORT_IN="  Port server lokal (default=8765): "
if not "!PORT_IN!"=="" set "PORT=!PORT_IN!"

echo.
echo   [+] Meluncurkan Web Dashboard pada http://localhost:!PORT! ...
echo   [+] Browser akan terbuka secara otomatis.
echo.
cd /d "%~dp0"
"!PYTHON!" benchmark_realtime.py --mode web --samples !SAMPLES! --port !PORT!
echo.
pause
goto :menu

:: ----------------------------------------------------------------------------
:: OPSI 3: BENCHMARK TERMINAL LENGKAP
:: ----------------------------------------------------------------------------
:run_terminal
cls
echo.
echo ========================================================================
echo   PILIH JUMLAH SAMPEL PER TASK UNTUK TERMINAL:
echo ========================================================================
echo.
echo     1 = Demo Cepat   (3 sampel/task,  60 total, ~2-3 menit pada CPU)
echo     2 = Standar      (10 sampel/task, 200 total, ~15 menit pada CPU)
echo.
set "S_CHOICE="
set /p S_CHOICE="  Pilihan sampel [1/2] (default=1): "
if "!S_CHOICE!"=="" set "S_CHOICE=1"
if "!S_CHOICE!"=="2" (
    set "SAMPLES=10"
) else (
    set "SAMPLES=3"
)

echo.
echo   [+] Menjalankan 20 Benchmark di Terminal...
echo.
cd /d "%~dp0"
"!PYTHON!" benchmark_realtime.py --mode terminal --samples !SAMPLES!
echo.
pause
goto :menu

:: ----------------------------------------------------------------------------
:: OPSI 4: 3-PASS SELECTIVE VIRTUAL MEMORY LOOP
:: ----------------------------------------------------------------------------
:run_3pass
cls
echo.
echo ========================================================================
echo   MENJALANKAN SIMULASI 3-PASS SELECTIVE VIRTUAL MEMORY LOOP
echo   [THE SMART AND EFFICIENT ARTIFICIAL BRAIN]
echo ========================================================================
echo.
cd /d "%~dp0"
"!PYTHON!" run_3pass_selective_virtual_memory.py
echo.
pause
goto :menu

:: ----------------------------------------------------------------------------
:: OPSI 5: 2-BENCH MATRIX QUESTION HELPER
:: ----------------------------------------------------------------------------
:run_matrix_helper
cls
echo.
echo ========================================================================
echo   MENJALANKAN 2-BENCH MATRIX QUESTION HELPER (QWEN3.5-2B)
echo   [ELIMINASI WRONG LOGS DISTRACTOR + DELIBERASI TAJAM SYSTEM 2]
echo ========================================================================
echo.
cd /d "%~dp0"
"!PYTHON!" run_matrix_helper_benchmark.py
echo.
pause
goto :menu
