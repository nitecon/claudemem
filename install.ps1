#Requires -Version 5.1
<#
.SYNOPSIS
    Install or upgrade agent-memory on Windows.
.DESCRIPTION
    Downloads the latest agent-memory release from GitHub and installs it
    to %USERPROFILE%\.agentic\bin\memory.exe. Adds the directory to the
    user's PATH if not already present.
#>

$ErrorActionPreference = "Stop"

$Repo = "nitecon/agent-memory"
$BinaryName = "memory.exe"
$InstallDir = Join-Path $env:USERPROFILE ".agentic\bin"

# --- Helpers ----------------------------------------------------------------

function Info($msg)  { Write-Host "[INFO]  $msg" -ForegroundColor Green }
function Warn($msg)  { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Fail($msg)  { Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

# --- Resolve latest version -------------------------------------------------

Info "Resolving latest release..."

try {
    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest" -UseBasicParsing
    $LatestTag = $release.tag_name
} catch {
    Fail "Could not determine latest release from GitHub: $_"
}

if (-not $LatestTag) {
    Fail "Could not determine latest release tag."
}

Info "Latest version: $LatestTag"

$ArchiveName = "agent-memory-windows-x86_64.zip"
$DownloadUrl = "https://github.com/$Repo/releases/download/$LatestTag/$ArchiveName"

# --- Check existing installation --------------------------------------------

$BinaryPath = Join-Path $InstallDir $BinaryName
if (Test-Path $BinaryPath) {
    try {
        $currentVersion = & $BinaryPath --version 2>$null
        Info "Existing installation found: $currentVersion"
    } catch {
        Info "Existing installation found (version unknown)"
    }
    Info "Upgrading to $LatestTag..."
} else {
    Info "No existing installation found. Installing fresh."
}

# --- Download and extract ---------------------------------------------------

$TmpDir = Join-Path $env:TEMP "agent-memory-install-$(Get-Random)"
New-Item -ItemType Directory -Path $TmpDir -Force | Out-Null

try {
    Info "Downloading $ArchiveName..."
    $archivePath = Join-Path $TmpDir $ArchiveName
    Invoke-WebRequest -Uri $DownloadUrl -OutFile $archivePath -UseBasicParsing

    Info "Extracting..."
    Expand-Archive -Path $archivePath -DestinationPath $TmpDir -Force

    # --- Install ----------------------------------------------------------------

    if (-not (Test-Path $InstallDir)) {
        New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    }

    Copy-Item -Path (Join-Path $TmpDir $BinaryName) -Destination $BinaryPath -Force
    Info "Installed $BinaryPath"

} finally {
    Remove-Item -Path $TmpDir -Recurse -Force -ErrorAction SilentlyContinue
}

# --- Add to PATH ------------------------------------------------------------

$userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($userPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("PATH", "$userPath;$InstallDir", "User")
    $env:PATH = "$env:PATH;$InstallDir"
    Info "Added $InstallDir to user PATH"
} else {
    Info "$InstallDir already in PATH"
}

# --- Done -------------------------------------------------------------------

Write-Host ""
Info "Installation complete!"
Write-Host ""
Write-Host "  Binary:  $BinaryPath"
Write-Host "  Version: $LatestTag"
Write-Host ""
Write-Host "Quick start:"
Write-Host "  memory store `"my first memory`" -m user -t `"test`""
Write-Host "  memory search `"first memory`""
Write-Host ""
Write-Host "Register as MCP server for Claude Code:"
Write-Host "  claude mcp add agent-memory -- `"$BinaryPath`" serve"
Write-Host ""
