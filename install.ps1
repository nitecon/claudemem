#Requires -Version 5.1
<#
.SYNOPSIS
    Install or upgrade agent-memory on Windows.
.DESCRIPTION
    Downloads the latest agent-memory release from GitHub and installs
    both bundled binaries (memory.exe + memory-dream.exe) to
    %USERPROFILE%\.agentic\bin\. Adds the directory to the user's PATH
    if not already present.

    Release 2 ships a single combined archive per platform that contains
    both binaries. If the companion binary (memory-dream.exe) is missing
    from a legacy archive this script warns and installs memory.exe
    alone rather than failing hard.
#>

$ErrorActionPreference = "Stop"

$Repo = "nitecon/agent-memory"
$Binaries = @("memory.exe", "memory-dream.exe")
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

# Release 2 asset name format: `agent-memory-<tag>-windows-x86_64.zip`.
$ArchiveName = "agent-memory-$LatestTag-windows-x86_64.zip"
$DownloadUrl = "https://github.com/$Repo/releases/download/$LatestTag/$ArchiveName"

# --- Check existing installation --------------------------------------------

$MemoryPath = Join-Path $InstallDir "memory.exe"
if (Test-Path $MemoryPath) {
    try {
        $currentVersion = & $MemoryPath --version 2>$null
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

    foreach ($bin in $Binaries) {
        $src = Join-Path $TmpDir $bin
        if (-not (Test-Path $src)) {
            if ($bin -eq "memory.exe") {
                Fail "Archive did not contain 'memory.exe'"
            }
            Warn "Archive did not contain '$bin' — skipping (older release?)"
            continue
        }
        $dest = Join-Path $InstallDir $bin
        Copy-Item -Path $src -Destination $dest -Force
        Info "Installed $dest"
    }

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
Write-Host "  memory:       $InstallDir\memory.exe"
Write-Host "  memory-dream: $InstallDir\memory-dream.exe"
Write-Host "  Version:      $LatestTag"
Write-Host ""
Write-Host "Quick start:"
Write-Host "  memory store `"my first memory`" -m user -t `"test`""
Write-Host "  memory search `"first memory`""
Write-Host ""
Write-Host "Compact the DB on a schedule (optional — run manually or via scheduled task):"
Write-Host "  memory-dream --pull          # one-time, fetch gemma3 weights"
Write-Host "  memory-dream --dry-run       # preview condensations + dedup"
Write-Host "  memory-dream                 # apply condensation + dedup"
Write-Host ""
Write-Host "Register as MCP server for Claude Code:"
Write-Host "  claude mcp add agent-memory -- `"$InstallDir\memory.exe`" serve"
Write-Host ""
