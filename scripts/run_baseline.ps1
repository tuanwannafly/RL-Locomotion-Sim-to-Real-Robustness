# scripts/run_baseline.ps1 — train a baseline policy on local GPU
# Usage: .\scripts\run_baseline.ps1 -Env Ant-v4 -Config configs/baseline_ant.yaml -Timesteps 2500000
[CmdletBinding()]
param(
    [string]$Env = "Ant-v4",
    [string]$Config = "configs/baseline_ant.yaml",
    [int]$Timesteps = 2500000,
    [string]$RunName = "baseline_ant",
    [string]$Device = "cuda"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
& .\.venv\Scripts\python.exe src\train.py --config $Config --env $Env --timesteps $Timesteps --run-name $RunName --device $Device