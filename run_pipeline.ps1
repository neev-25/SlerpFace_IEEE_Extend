# SlerpFace Pipeline - Windows PowerShell Script
# Run from: D:\Coding\SlerpFace\recognition\
# Usage: .\tasks\slerpface\run_pipeline.ps1

param(
    [string]$DataRoot = ".\tasks\slerpface\datasets",
    [string]$Key = "CFP",
    [float]$Alpha = 0.9,
    [float]$DropRate = 0.5,
    [int]$GroupSize = 16,
    [int]$BatchSize = 64,
    [string]$GpuIds = "0"
)

$ErrorActionPreference = "Stop"

Write-Host "=== SlerpFace Pipeline ===" -ForegroundColor Cyan
Write-Host "Working directory: $(Get-Location)"

# Step 1: Feature Extraction
Write-Host "`n[Step 1/3] Extracting face templates..." -ForegroundColor Yellow
python -m tasks.slerpface.extract_template.verification `
    --model_path=.\tasks\slerpface\ckpt\Backbone_Epoch_24_checkpoint.pth `
    --data_root=$DataRoot `
    --backbone=IR_50 `
    --batch_size=$BatchSize `
    --output_dir=.\tasks\slerpface\templates `
    --gpu_ids=$GpuIds

if ($LASTEXITCODE -ne 0) { Write-Error "Feature extraction failed!"; exit 1 }
Write-Host "[Step 1/3] Done." -ForegroundColor Green

# Step 2: Template Encryption
Write-Host "`n[Step 2/3] Encrypting templates..." -ForegroundColor Yellow
python -m tasks.slerpface.encrypt_template.enrollment `
    --templates_folder .\tasks\slerpface\templates `
    --key $Key `
    --alpha $Alpha `
    --drop_rate $DropRate `
    --group_size $GroupSize

if ($LASTEXITCODE -ne 0) { Write-Error "Template encryption failed!"; exit 1 }
Write-Host "[Step 2/3] Done." -ForegroundColor Green

# Step 3: Template Matching
Write-Host "`n[Step 3/3] Matching templates..." -ForegroundColor Yellow
python -m tasks.slerpface.match_template.gen_sim `
    --templates_folder .\tasks\slerpface\templates `
    --key $Key `
    --alpha $Alpha `
    --group_size $GroupSize

if ($LASTEXITCODE -ne 0) { Write-Error "Template matching failed!"; exit 1 }
Write-Host "[Step 3/3] Done." -ForegroundColor Green

Write-Host "`n=== SlerpFace Pipeline Complete ===" -ForegroundColor Cyan
