# Salin model TFLite + labels hasil export ke assets aplikasi Android.
param(
    [string]$Model = "training\exported\freshness_mnv4s_r224_fp16.tflite",
    [string]$Labels = "training\exported\labels.txt"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Assets = Join-Path $Root "android\app\src\main\assets"
New-Item -ItemType Directory -Force -Path $Assets | Out-Null

Copy-Item (Join-Path $Root $Model) (Join-Path $Assets "freshness_model.tflite") -Force
Copy-Item (Join-Path $Root $Labels) (Join-Path $Assets "labels.txt") -Force
Write-Host "Model & labels disalin ke $Assets"
