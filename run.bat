@echo off
REM ╔══════════════════════════════════════════════════════════════════════╗
REM ║  KYC Verification System — Windows Launcher (Python 3.11 / Win x64) ║
REM ╚══════════════════════════════════════════════════════════════════════╝

cd /d "%~dp0"

echo [1/6] Checking Python version...
py -3.11 --version 2>NUL
IF ERRORLEVEL 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause & exit /b 1
)

REM ── Virtual environment setup ──────────────────────────────────────────
echo [2/6] Setting up isolated virtual environment (.venv_kyc)...

REM Validate the venv is healthy (not relocated from another machine)
IF EXIST ".venv_kyc\Scripts\python.exe" (
    .venv_kyc\Scripts\python.exe --version >NUL 2>&1
    IF ERRORLEVEL 1 (
        echo       Broken or relocated venv detected -- removing and recreating...
        rmdir /s /q .venv_kyc
    )
)

IF NOT EXIST ".venv_kyc\Scripts\python.exe" (
    echo       Creating new virtual environment...
    py -3.11 -m venv .venv_kyc
    IF ERRORLEVEL 1 (
        echo ERROR: Failed to create virtual environment.
        pause & exit /b 1
    )
    echo       Virtual environment created successfully.
    echo       Upgrading PIP...
    .venv_kyc\Scripts\python.exe -m pip install --quiet --upgrade pip
) ELSE (
    echo       Virtual environment already exists and is healthy -- skipping creation.
)

REM Point all pip/python commands to the venv
SET VENV_PIP=.venv_kyc\Scripts\pip.exe
SET VENV_PY=.venv_kyc\Scripts\python.exe

REM ── Check if packages are already installed (fast-path on re-runs) ─────
%VENV_PY% -c "import customtkinter, cv2, paddle, insightface, torch" 2>NUL
IF NOT ERRORLEVEL 1 (
    echo       All packages already installed -- skipping install steps.
    GOTO LAUNCH
)

REM ── Fresh install ──────────────────────────────────────────────────────
echo [3/6] Installing GUI + CV + OCR packages...
%VENV_PIP% install --quiet --no-cache-dir ^
    customtkinter==5.2.2 ^
    "CTkMessagebox==2.5" ^
    "Pillow==10.1.0" ^
    "opencv-python==4.8.1.78" ^
    "paddlepaddle==2.6.2" ^
    "paddleocr==2.8.1" ^
    "numpy==1.26.4" ^
    "protobuf==3.20.2"
IF ERRORLEVEL 1 (
    echo ERROR: Failed to install GUI/CV/OCR packages.
    pause & exit /b 1
)

echo [4/6] Installing InsightFace pre-built wheel (Python 3.11)...
REM Try local wheel first, fall back to remote URL
IF EXIST "insightface-0.7.3-cp311-cp311-win_amd64.whl" (
    echo       Using local wheel file...
    %VENV_PIP% install --quiet --no-cache-dir "insightface-0.7.3-cp311-cp311-win_amd64.whl"
) ELSE (
    echo       Downloading wheel from GitHub...
    %VENV_PIP% install --quiet --no-cache-dir "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp311-cp311-win_amd64.whl"
)
IF ERRORLEVEL 1 (
    echo WARNING: pip reported a dependency conflict for InsightFace. Proceeding to fix dependencies...
)

REM Pin numpy/protobuf/onnx back after insightface deps may upgrade them:
%VENV_PIP% install --quiet --no-cache-dir "onnx==1.14.1" "numpy==1.26.4" "protobuf==3.20.2" "onnxruntime==1.16.3"

echo [5/6] Installing Deep Learning packages (PyTorch + timm + XGBoost)...
%VENV_PIP% install --quiet --no-cache-dir ^
    "torch==2.1.2" ^
    "torchvision==0.16.2" ^
    "timm==0.9.7" ^
    "xgboost==2.0.3" ^
    "scikit-learn==1.3.2" ^
    "scipy==1.11.4"
IF ERRORLEVEL 1 (
    echo ERROR: Failed to install Deep Learning packages.
    pause & exit /b 1
)

:LAUNCH
echo [6/6] Launching KYC Verification System...
%VENV_PY% main.py

pause
