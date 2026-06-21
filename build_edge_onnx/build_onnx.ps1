# Build the CalEyeZ ONNX edge demo into a standalone Windows .exe (CPU-only, torch-free, no GPU).
# Run from PowerShell:  cd build_edge_onnx ; .\build_onnx.ps1
# Output: build_edge_onnx\dist\CalEyeZ\CalEyeZ.exe  (zip the CalEyeZ folder to copy to any laptop)
#
# Prereq: the ONNX models must exist (build_edge_onnx\models\*.onnx). If not, generate them first:
#   py -3 scripts\edge\onnx_export_and_check.py
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\models\global.onnx") -or -not (Test-Path ".\models\israeli.onnx")) {
    Write-Host "[onnx-edge] ONNX models missing. Run:  py -3 scripts\edge\onnx_export_and_check.py" -ForegroundColor Yellow
    exit 1
}

$venv = ".\.venv_onnx"
$py   = "$venv\Scripts\python.exe"
if (-not (Test-Path $venv)) {
    Write-Host "[onnx-edge] creating venv..."
    $launcher = (Get-Command py -ErrorAction SilentlyContinue)
    if ($launcher) { py -3.11 -m venv $venv } else { python -m venv $venv }
}

& $py -m pip install --upgrade pip
# Lean, CPU-only, torch-free runtime: onnxruntime instead of torch/ultralytics.
& $py -m pip install onnxruntime opencv-python pillow xgboost scikit-learn pandas numpy matplotlib bleak requests google-generativeai pyinstaller

Write-Host "[onnx-edge] running PyInstaller..."
& $py -m PyInstaller caleyez_onnx.spec --noconfirm --clean

Write-Host "[onnx-edge] DONE -> dist\CalEyeZ\CalEyeZ.exe" -ForegroundColor Green
Write-Host "[onnx-edge] Copy the whole dist\CalEyeZ folder to the target laptop and run CalEyeZ.exe." -ForegroundColor Green
