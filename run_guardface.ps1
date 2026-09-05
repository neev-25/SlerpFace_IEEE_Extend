# GuardFace Execution Helper for Windows PowerShell
param(
    [string]$Mode = "demo" # "demo", "cli", "eval", "test"
)

$PythonExe = "C:\Users\meetk\anaconda3\envs\slerpface\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "=== GuardFace: Anti-Spoofing & SlerpFace Security Suite ===" -ForegroundColor Cyan
Write-Host "Using Python: $PythonExe"

# Generate samples if not present
if (-not (Test-Path ".\samples\genuine_personA_1.jpg")) {
    Write-Host "`n[Setup] Generating synthetic test presentations in .\samples\..." -ForegroundColor Yellow
    & $PythonExe .\create_demo_samples.py
}

switch ($Mode.ToLower()) {
    "test" {
        Write-Host "`n[Mode: Unit Tests] Running subsystem tests..." -ForegroundColor Green
        & $PythonExe .\tests\test_antispoof.py
    }
    "eval" {
        Write-Host "`n[Mode: Benchmark] Running ISO/IEC 30107-3 PAD benchmark..." -ForegroundColor Green
        & $PythonExe .\evaluate_antispoof.py
    }
    "cli" {
        Write-Host "`n[Mode: CLI Test] Running 1:1 dual-tier verification..." -ForegroundColor Green
        Write-Host "1. Testing Genuine vs Genuine (Live user):" -ForegroundColor White
        & $PythonExe .\verify_with_antispoof.py --img1 .\samples\genuine_personA_1.jpg --img2 .\samples\genuine_personA_2.jpg

        Write-Host "`n2. Testing Genuine vs Screen Replay Attack (Spoof attempt):" -ForegroundColor White
        & $PythonExe .\verify_with_antispoof.py --img1 .\samples\genuine_personA_1.jpg --img2 .\samples\spoof_screen_replay.jpg
    }
    "demo" {
        Write-Host "`n[Mode: Web Dashboard] Launching Streamlit Web App..." -ForegroundColor Green
        & $PythonExe -m streamlit run .\app_demo.py
    }
    default {
        Write-Host "Unknown mode: $Mode. Available modes: demo, cli, eval, test" -ForegroundColor Red
    }
}
