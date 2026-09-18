# Windows environment setup: Python venv + PyTorch (CUDA 13.0) + verify LibTorch CMake path.
# Run from the repository root:  powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".venv")) {
    Write-Host "Creating venv (.venv)..."
    py -3.14 -m venv .venv
}
$Py = Join-Path $Root ".venv\Scripts\python.exe"
& $Py -m pip install --upgrade pip
& $Py -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu130

& $Py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('cmake prefix', torch.utils.cmake_prefix_path)"

Write-Host ""
Write-Host "Toolchain check:"
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
} else {
    Write-Warning "vswhere not found; install Visual Studio with 'Desktop development with C++'."
}
cmake --version | Select-Object -First 1
nvcc --version | Select-Object -Last 1

Write-Host ""
Write-Host "Done. Configure the C++ engine with:"
Write-Host "  cmake -S . -B build -G 'Visual Studio 18 2026' -A x64"
Write-Host "  cmake --build build --config Release"
