# Clone source TensorFlow (hanya untuk membangun TensorFlow Lite C++).
# Tag HARUS sama dengan versi tensorflow di training/requirements.txt.
param(
    [string]$Tag = "v2.17.1"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Dest = Join-Path $Root "third_party\tensorflow"

$StbDir = Join-Path $Root "third_party\stb"
New-Item -ItemType Directory -Force -Path (Split-Path $Dest) | Out-Null

# stb_image.h: dipakai CLI desktop classify_image
if (-not (Test-Path (Join-Path $StbDir "stb_image.h"))) {
    git clone --depth 1 https://github.com/nothings/stb.git $StbDir
    if ($LASTEXITCODE -ne 0) { throw "git clone stb gagal" }
}

if (Test-Path (Join-Path $Dest "tensorflow\lite\CMakeLists.txt")) {
    Write-Host "Source TensorFlow sudah ada di $Dest"
    exit 0
}

git -c core.longpaths=true clone --depth 1 --branch $Tag https://github.com/tensorflow/tensorflow.git $Dest
if ($LASTEXITCODE -ne 0) { throw "git clone gagal" }
Write-Host "Selesai: $Dest ($Tag)"
