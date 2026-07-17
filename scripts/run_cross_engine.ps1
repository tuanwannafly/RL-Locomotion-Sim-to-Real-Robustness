# scripts/run_cross_engine.ps1 — eval MuJoCo-trained policy in PyBullet
[CmdletBinding()]
param(
    [string]$Run = "dr_ant",
    [string]$Env = "AntBulletEnv-v0",
    [string]$OutCsv = "experiments/results/sprint5_cross_engine.csv"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
& .\.venv\Scripts\python.exe src\cross_engine_eval.py --run $Run --env $Env --out-csv $OutCsv