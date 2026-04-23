param(
    [string]$RepoRoot = "C:\dev\Personal-Trading-Portfolio-Project",
    [string]$RunTag = "",
    [switch]$DryRun,
    [string]$RebalanceCalendar = "strategy"
)

$ErrorActionPreference = "Stop"

if (-not $RunTag) {
    $RunTag = Get-Date -Format "yyyyMMdd_HHmmss"
}

$repoPath = [System.IO.Path]::GetFullPath($RepoRoot)
$logsRoot = Join-Path $repoPath "logs"
$metaLog = Join-Path $logsRoot ("paper_shadow_{0}_meta.log" -f $RunTag)
$stdoutLog = Join-Path $logsRoot ("paper_shadow_{0}.log" -f $RunTag)
$stderrLog = Join-Path $logsRoot ("paper_shadow_{0}.stderr.log" -f $RunTag)

New-Item -ItemType Directory -Force -Path $logsRoot | Out-Null

function Write-Meta {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $metaLog -Value $line
    Write-Output $line
}

# Task Scheduler registration example:
# schtasks /Create /SC DAILY /TN "PaperShadowDaily" /TR "powershell -ExecutionPolicy Bypass -File C:\dev\Personal-Trading-Portfolio-Project\automation\run_paper_shadow.ps1" /ST 16:45 /F

Set-Location $repoPath

$asOfDate = Get-Date -Format "yyyy-MM-dd"
$weekday = (Get-Date).DayOfWeek
if ($weekday -in @("Saturday", "Sunday")) {
    Write-Meta "Skipping weekend run for $asOfDate"
    exit 0
}

$checkScript = @"
import json
import sys
from pathlib import Path
repo_root = Path(r"$repoPath")
sys.path.insert(0, str(repo_root))
from src.paper.loaders import inspect_rebalance_day
payload = inspect_rebalance_day(as_of_date="$asOfDate", repo_root=repo_root)
print(json.dumps(payload))
"@

$checkScriptPath = Join-Path $env:TEMP ("paper_shadow_check_{0}.py" -f $RunTag)
Set-Content -Path $checkScriptPath -Value $checkScript -Encoding UTF8

$rebalanceJson = & python $checkScriptPath 2>> $stderrLog
$checkExitCode = $LASTEXITCODE
Remove-Item $checkScriptPath -ErrorAction SilentlyContinue
if ($checkExitCode -ne 0) {
    Write-Meta "Rebalance-day probe failed"
    exit $checkExitCode
}

$rebalanceInfo = $rebalanceJson | ConvertFrom-Json
if (-not $rebalanceInfo.is_rebalance_day) {
    Write-Meta "Skipping non-rebalance day $asOfDate (calendar=$RebalanceCalendar)"
    exit 0
}

$command = @(".\run_paper_shadow.py", "--as-of-date", $asOfDate)
if ($DryRun) {
    $command += "--dry-run"
}

Write-Meta "Repo root: $repoPath"
Write-Meta "Run tag: $RunTag"
Write-Meta "DryRun: $DryRun"
Write-Meta "RebalanceCalendar: $RebalanceCalendar"
Write-Meta ("Command: python {0}" -f ($command -join " "))

& python @command 1>> $stdoutLog 2>> $stderrLog
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Meta "Completed successfully"
} else {
    Write-Meta "Failed with exit code $exitCode"
}

exit $exitCode
