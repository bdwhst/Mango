# Configure, build and test on Windows.
#   powershell -ExecutionPolicy Bypass -File scripts\build.ps1            # core + tests, no LibTorch
#   powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Torch     # with LibTorch (CUDA)
param([switch]$Torch, [switch]$NoTest)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$preset = if ($Torch) { "windows-cuda" } else { "windows-msvc" }
cmake --preset $preset
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cmake --build --preset $preset
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (-not $NoTest) {
  ctest --preset $preset
  exit $LASTEXITCODE
}
