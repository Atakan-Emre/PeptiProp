param(
    [string]$Config = "configs/ablation_generated/ablation_mpnn_f2_l2_c0_full.yaml",
    [switch]$SkipSplit,
    [switch]$SkipPairs,
    [switch]$SkipLeakageTests
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUNBUFFERED = "1"

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $root ("outputs\logs\single_full_training_" + $ts)
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Write-Host ("LOG DIR: {0}" -f $logDir) -ForegroundColor Yellow

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Command
    )

    $logFile = Join-Path $logDir ($Name + ".log")
    Write-Host ""
    Write-Host ("[{0}] RUN: {1}" -f $Name, $Command) -ForegroundColor Cyan

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
    param([Parameter(Mandatory = $true)][string]$Name)

    $logFile = Join-Path $logDir ($Name + ".log")
    Write-Host ""
    Write-Host ("[{0}] RUN: leakage guard tests" -f $Name) -ForegroundColor Cyan

    & cmd /c "python -m pytest --version >nul 2>&1"
    if ($LASTEXITCODE -eq 0) {
        Write-Host ("[{0}] using pytest" -f $Name) -ForegroundColor DarkCyan
        & cmd /c "python -m pytest tests/test_baseline_leakage_guards.py -q 2>&1" | Tee-Object -FilePath $logFile
    }
    else {
        Write-Host ("[{0}] pytest not found, using fallback runner" -f $Name) -ForegroundColor DarkYellow
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

Invoke-Step "00_annotate" "python scripts/annotate_interface_pocket.py --canonical data/canonical"
if (-not $SkipSplit) {
    Invoke-Step "01_split" "python scripts/build_pdb_level_splits.py --canonical data/canonical --propedia-meta data/raw/propedia --out data/canonical/splits --seed 42"
}
else {
    Write-Host "[01_split] SKIPPED" -ForegroundColor Yellow
}

if (-not $SkipPairs) {
    Invoke-Step "02_pairs" "python scripts/generate_negative_pairs.py --canonical data/canonical --splits data/canonical/splits --output data/canonical/pairs --seed 42"
}
else {
    Write-Host "[02_pairs] SKIPPED" -ForegroundColor Yellow
}

if (-not $SkipLeakageTests) {
    Invoke-Leakage-Tests "03_leakage_tests"
}
else {
    Write-Host "[03_leakage_tests] SKIPPED" -ForegroundColor Yellow
}

Invoke-Step "04_train" ("python scripts/train_scoring_model.py --config " + $Config)

Write-Host ""
Write-Host ("DONE. Logs: {0}" -f $logDir) -ForegroundColor Yellow
