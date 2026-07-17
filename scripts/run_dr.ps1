# scripts/run_dr.ps1 — train a domain-randomized policy
[CmdletBinding()]
param(
    [string]$Env = "Ant-v4",
    [string]$Config = "configs/dr_ant.yaml",
    [int]$Timesteps = 2500000,
    [string]$RunName = "dr_ant",
    [string]$Device = "cuda"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
& .\.venv\Scripts\python.exe src\train.py --config $Config --env $Env --timesteps $Timesteps --run-name $RunName --device $Device