# scripts/run_eval.ps1 — run robustness eval harness
[CmdletBinding()]
param(
    [string]$BaselineRun = "baseline_ant",
    [string]$DrRun = "dr_ant",
    [string]$Env = "Ant-v4",
    [string]$OutCsv = "experiments/results/sprint4_robustness_ant.csv",
    [int]$EpisodesPerCell = 30
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
& .\.venv\Scripts\python.exe src\eval.py `
    --env $Env `
    --baseline-run $BaselineRun `
    --dr-run $DrRun `
    --out-csv $OutCsv `
    --episodes-per-cell $EpisodesPerCell