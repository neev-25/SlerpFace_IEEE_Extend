# GuardFace launcher. Paths are rooted at this script, independent of caller CWD.
param(
    [string]$Mode = "demo",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $workspacePython = Join-Path $PSScriptRoot ".venv/Scripts/python.exe"
    $PythonExe = if (Test-Path -LiteralPath $workspacePython -PathType Leaf) { $workspacePython } else { "python" }
}
$selectedMode = $Mode.ToLowerInvariant()
if ($selectedMode -notin @("demo", "cli", "eval", "test")) {
    [Console]::Error.WriteLine("Unknown mode '$Mode'. Available modes: demo, cli, eval, test.")
    exit 2
}

$exitCode = 2
Push-Location -LiteralPath $PSScriptRoot
try {
    Write-Host "GuardFace research demonstration - Python: $PythonExe" -ForegroundColor Cyan
    switch ($selectedMode) {
        "test" {
            & $PythonExe -B -m pytest -p no:cacheprovider (Join-Path $PSScriptRoot "tests")
            $exitCode = $LASTEXITCODE
        }
        "eval" {
            Write-Host "Synthetic heuristic benchmark; this does not establish standards compliance."
            & $PythonExe -B (Join-Path $PSScriptRoot "evaluate_antispoof.py")
            $exitCode = $LASTEXITCODE
        }
        "cli" {
            $reference = Join-Path $PSScriptRoot "samples/genuine_personA_1.jpg"
            $probes = @("genuine_personA_2.jpg", "spoof_screen_replay.jpg")
            $exitCode = 0
            foreach ($probe in $probes) {
                $probePath = Join-Path $PSScriptRoot "samples/$probe"
                if (-not (Test-Path -LiteralPath $reference -PathType Leaf) -or
                    -not (Test-Path -LiteralPath $probePath -PathType Leaf)) {
                    throw "Bundled samples are missing. Run create_demo_samples.py explicitly to recreate them."
                }
                & $PythonExe -B (Join-Path $PSScriptRoot "verify_with_antispoof.py") --demo --img1 $reference --img2 $probePath
                $commandExitCode = $LASTEXITCODE
                if ($commandExitCode -gt $exitCode) { $exitCode = $commandExitCode }
                if ($commandExitCode -gt 1) { break }
            }
        }
        "demo" {
            & $PythonExe -B -m streamlit run (Join-Path $PSScriptRoot "app_demo.py")
            $exitCode = $LASTEXITCODE
        }
    }
}
catch {
    [Console]::Error.WriteLine("GuardFace launcher failed: $($_.Exception.Message)")
    $exitCode = 2
}
finally {
    Pop-Location
}
exit $exitCode
