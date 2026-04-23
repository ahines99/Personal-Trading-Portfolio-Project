param(
    [string]$RepoRoot = "C:\dev\Personal-Trading-Portfolio-Project",
    [ValidateSet("phase-a", "phase-b", "auto")]
    [string]$Mode = "auto",
    [string]$Config = "",
    [string]$AsOfDate = "",
    [string]$BundleDir = "",
    [string]$OutputDir = "",
    [string]$RunTag = "",
    [switch]$ChainApprovedPhaseB
)

$ErrorActionPreference = "Stop"

if (-not $RunTag) {
    $RunTag = Get-Date -Format "yyyyMMdd_HHmmss"
}

$repoPath = [System.IO.Path]::GetFullPath($RepoRoot)
$logsRoot = Join-Path $repoPath "logs"
$metaLog = Join-Path $logsRoot ("paper_daily_{0}_meta.log" -f $RunTag)
$stdoutLog = Join-Path $logsRoot ("paper_daily_{0}.log" -f $RunTag)
$stderrLog = Join-Path $logsRoot ("paper_daily_{0}.stderr.log" -f $RunTag)

New-Item -ItemType Directory -Force -Path $logsRoot | Out-Null

function Write-Meta {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $metaLog -Value $line
    Write-Output $line
}

function Get-ConfigValue {
    param(
        [string]$ConfigPath,
        [string]$Key
    )
    if (-not $ConfigPath -or -not (Test-Path -LiteralPath $ConfigPath)) {
        return $null
    }

    $extension = [System.IO.Path]::GetExtension($ConfigPath)
    if ($extension -eq ".json") {
        try {
            $json = Get-Content -Raw -LiteralPath $ConfigPath | ConvertFrom-Json
            $value = $json.$Key
            if ($null -ne $value) {
                return [string]$value
            }
        } catch {
            return $null
        }
        return $null
    }

    foreach ($rawLine in Get-Content -LiteralPath $ConfigPath) {
        if ($rawLine -match "^\s*$Key\s*:\s*(.+?)\s*$") {
            return $Matches[1].Trim("'`"")
        }
    }
    return $null
}

function Resolve-KillSwitchPath {
    param(
        [string]$RepoPath,
        [string]$Config
    )
    $candidate = "paper_trading/state/KILL_SWITCH"
    if ($Config) {
        $configPath = $Config
        if (-not [System.IO.Path]::IsPathRooted($configPath)) {
            $configPath = Join-Path $RepoPath $configPath
        }
        $configPath = [System.IO.Path]::GetFullPath($configPath)
        $configuredPath = Get-ConfigValue -ConfigPath $configPath -Key "kill_switch_path"
        if ($configuredPath) {
            $candidate = $configuredPath
        }
    }
    if ([System.IO.Path]::IsPathRooted($candidate)) {
        return [System.IO.Path]::GetFullPath($candidate)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoPath $candidate))
}

# Task Scheduler registration example:
# schtasks /Create /SC DAILY /TN "PaperTradingDaily" /TR "powershell -ExecutionPolicy Bypass -File C:\dev\Personal-Trading-Portfolio-Project\automation\run_paper_daily.ps1 -Mode auto" /ST 16:45 /F

Set-Location $repoPath

$command = @(".\run_paper_daily.py", "--mode", $Mode)
if ($Config) {
    $command += @("--config", $Config)
}
if ($AsOfDate) {
    $command += @("--as-of-date", $AsOfDate)
}
if ($BundleDir) {
    $command += @("--bundle-dir", $BundleDir)
}
if ($OutputDir) {
    $command += @("--output-dir", $OutputDir)
}
if ($ChainApprovedPhaseB) {
    $command += @("--chain-approved-phase-b")
}

Write-Meta "Repo root: $repoPath"
Write-Meta "Run tag: $RunTag"
Write-Meta "Mode: $Mode"
if ($Config) {
    Write-Meta "Config: $Config"
}
else {
    Write-Meta "Config: <default>"
}
$killSwitchPath = Resolve-KillSwitchPath -RepoPath $repoPath -Config $Config
Write-Meta "Kill-switch path: $killSwitchPath"
if (Test-Path -LiteralPath $killSwitchPath) {
    Write-Meta "Kill switch engaged before runner start: $killSwitchPath"
    exit 1
}
Write-Meta ("Command: python {0}" -f ($command -join " "))

& python @command 1>> $stdoutLog 2>> $stderrLog
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Meta "Completed successfully"
} elseif ($exitCode -eq 10) {
    Write-Meta "Completed awaiting approval"
} elseif ($exitCode -eq 11) {
    Write-Meta "Completed without Phase B because bundle is not approved"
} else {
    Write-Meta "Failed with exit code $exitCode"
}

exit $exitCode
