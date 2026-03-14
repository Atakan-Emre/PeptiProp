@echo off
echo ====================================
echo PeptidQuantum - Training Pipeline
echo ====================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found
    pause
    exit /b 1
)

REM Step 1: Prepare combined data
echo Step 1: Preparing data...
python scripts\prepare_data.py
if errorlevel 1 (
    echo ERROR: Data preparation failed
    pause
    exit /b 1
)

REM Step 2: Train model
echo.
echo Step 2: Training model...
python train.py --data data/processed/geppri_all_pairs.jsonl --epochs 100 --output-dir models/trained

pause
