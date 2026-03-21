$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = "1"

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $root ("outputs\logs\final_ablation_" + $ts)
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Command
    )

    $logFile = Join-Path $logDir ($Name + ".log")
    Write-Host ""
    Write-Host ("[{0}] RUN: {1}" -f $Name, $Command) -ForegroundColor Cyan

    # Stream live output to console AND persist to log.
    # cmd /c + 2>&1 avoids native stderr breaking PowerShell error handling.
    & cmd /c "$Command 2>&1" | Tee-Object -FilePath $logFile
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Host ""
        Write-Host ("[{0}] FAILED (exit={1}) -> {2}" -f $Name, $exitCode, $logFile) -ForegroundColor Red
        exit $exitCode
    }

    Write-Host ("[{0}] OK -> {1}" -f $Name, $logFile) -ForegroundColor Green
}

function Invoke-Leakage-Tests {
    param(
        [Parameter(Mandatory = $true)][string]$Name
    )

    $logFile = Join-Path $logDir ($Name + ".log")
    Write-Host ""
    Write-Host ("[{0}] RUN: leakage guard tests" -f $Name) -ForegroundColor Cyan

    # Prefer pytest if available; fallback to unittest runner script.
    & cmd /c "python -m pytest --version >nul 2>&1"
    if ($LASTEXITCODE -eq 0) {
        Write-Host ("[{0}] using pytest" -f $Name) -ForegroundColor DarkCyan
        & cmd /c "python -m pytest tests/test_baseline_leakage_guards.py -q 2>&1" | Tee-Object -FilePath $logFile
    }
    else {
        Write-Host ("[{0}] pytest not found, using unittest fallback" -f $Name) -ForegroundColor DarkYellow
        & cmd /c "python tests/test_baseline_leakage_guards.py 2>&1" | Tee-Object -FilePath $logFile
    }

    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-Host ""
        Write-Host ("[{0}] FAILED (exit={1}) -> {2}" -f $Name, $exitCode, $logFile) -ForegroundColor Red
        exit $exitCode
    }

    Write-Host ("[{0}] OK -> {1}" -f $Name, $logFile) -ForegroundColor Green
}

Invoke-Step "01_split" "python scripts/build_pdb_level_splits.py --canonical data/canonical --propedia-meta data/raw/propedia --out data/canonical/splits --seed 42"
Invoke-Step "02_pairs" "python scripts/generate_negative_pairs.py --canonical data/canonical --splits data/canonical/splits --output data/canonical/pairs --seed 42"
Invoke-Leakage-Tests "03_leakage_tests"
Invoke-Step "04_ablation_smoke" "python scripts/run_classical_ablation.py --smoke-only --smoke-epochs 8 --smoke-patience 4"
Invoke-Step "05_ablation_full_200ep" "python scripts/run_classical_ablation.py --full-epochs 200 --full-patience 20 --smoke-epochs 8 --smoke-patience 4 --finalists-per-family 1 --full-subset-train 60000 --full-subset-val 20000 --full-subset-test 20000"

Write-Host ""
Write-Host ("DONE. Logs: {0}" -f $logDir) -ForegroundColor Yellow
Write-Host ("Best-summary: {0}" -f (Join-Path $root "outputs\training\peptidquantum_v0_1_final_best_classical\ablation_summary.csv")) -ForegroundColor Yellow
