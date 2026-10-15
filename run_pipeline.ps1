# Compatibility notice for the unavailable original multi-stage research pipeline.
param(
    [string]$DataRoot = "",
    [string]$Key = "CFP",
    [float]$Alpha = 0.9,
    [float]$DropRate = 0.5,
    [int]$GroupSize = 16,
    [int]$BatchSize = 64,
    [string]$GpuIds = "0"
)

# Fail before extraction or file writes: this checkout has no enrollment/matching
# packages, so running the first stage alone would leave misleading partial output.
$launcher = Join-Path $PSScriptRoot "run_guardface.ps1"
[Console]::Error.WriteLine("The original three-stage dataset pipeline is unavailable: encrypt_template.enrollment and match_template.gen_sim are not included in this checkout.")
[Console]::Error.WriteLine("For the supported research demo, run: & '$launcher' -Mode cli")
[Console]::Error.WriteLine("For feature extraction only, from '$PSScriptRoot' run: python -m extract_template.verification --help. Supply a compatible checkpoint and external torchkit implementation; extraction does not enroll credentials.")
exit 2
