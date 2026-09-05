#Requires -Version 5.1
<#
.SYNOPSIS
    One-time developer build script for the local URL ingestion helper
    (local_helper/). Produces a standalone Windows application under
    dist/ that needs neither VS Code nor a terminal to run afterward --
    see docs/local_url_helper.md.

    This script is for DEVELOPMENT/BUILD use only. Once
    "Raza Audio URL Helper.exe" has been built, it is used by double-
    clicking it -- this script is never needed again unless the helper's
    code changes and a new build is required.

.DESCRIPTION
    1. Creates a DEDICATED build virtual environment (.venv-helper-build)
       and installs ONLY requirements-helper.txt into it -- deliberately
       NOT the main project .venv, so PyInstaller never bundles
       onnxruntime/streamlit/torch/etc. into the helper executable.
    2. Locates or downloads a pinned cloudflared.exe release into
       local_helper/tools/.
    3. Verifies ffmpeg.exe/ffprobe.exe are available (on PATH already, or
       already present in local_helper/tools/) and copies them into
       local_helper/tools/ if found elsewhere on this machine.
    4. Runs PyInstaller against local_helper/launcher.py.
    5. Copies local_helper/tools/ next to the built executable.
    6. Runs a quick smoke test against the built executable's underlying
       package (imports only -- does not open the GUI or start a real
       tunnel).
#>

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$BuildVenv = Join-Path $RepoRoot ".venv-helper-build"
$ToolsDir = Join-Path $RepoRoot "local_helper\tools"
$DistDir = Join-Path $RepoRoot "dist\Raza Audio URL Helper"

# Pin an exact cloudflared release -- never track "latest" (this
# project's policy for every vendored/downloaded third-party binary).
# Verified against https://api.github.com/repos/cloudflare/cloudflared/releases/latest
# at the time this was pinned.
$CloudflaredVersion = "2026.8.3"
$CloudflaredUrl = "https://github.com/cloudflare/cloudflared/releases/download/$CloudflaredVersion/cloudflared-windows-amd64.exe"

Write-Host "=== Step 1: build virtual environment ==="
if (-not (Test-Path $BuildVenv)) {
    python -m venv $BuildVenv
}
$BuildPython = Join-Path $BuildVenv "Scripts\python.exe"
& $BuildPython -m pip install --upgrade pip | Out-Null
& $BuildPython -m pip install -r (Join-Path $RepoRoot "requirements-helper.txt") pyinstaller
# --no-deps: makes `audio_deepfake_detector` (local_helper/youtube.py's
# dependency chain via app.analysis.video_url -> media_ffmpeg ->
# audio_deepfake_detector.preprocessing.audio_loader) a real, properly
# resolvable installed package for both PyInstaller's analysis AND this
# script's own smoke test below -- WITHOUT installing its heavyweight
# base dependencies (torch/torchaudio/transformers/etc, see
# pyproject.toml), which are already satisfied by the lighter
# requirements-helper.txt versions (numpy/soundfile/pydantic) or not
# needed by the helper at all.
& $BuildPython -m pip install -e "$RepoRoot" --no-deps

Write-Host "=== Step 2: locate or obtain cloudflared.exe ==="
New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
$CloudflaredDest = Join-Path $ToolsDir "cloudflared.exe"
if (Test-Path $CloudflaredDest) {
    Write-Host "cloudflared.exe already present in local_helper/tools/ -- skipping download."
} else {
    $ExistingCloudflared = Get-Command cloudflared.exe -ErrorAction SilentlyContinue
    if ($ExistingCloudflared) {
        Write-Host "Found cloudflared.exe on PATH at $($ExistingCloudflared.Source) -- copying it in."
        Copy-Item $ExistingCloudflared.Source $CloudflaredDest
    } else {
        Write-Host "Downloading pinned cloudflared $CloudflaredVersion from $CloudflaredUrl"
        Write-Host "(This downloads an official Cloudflare binary -- verify the URL/version above before running unattended.)"
        Invoke-WebRequest -Uri $CloudflaredUrl -OutFile $CloudflaredDest
    }
}

Write-Host "=== Step 3: verify ffmpeg/ffprobe ==="
foreach ($tool in @("ffmpeg.exe", "ffprobe.exe")) {
    $dest = Join-Path $ToolsDir $tool
    if (Test-Path $dest) {
        Write-Host "$tool already present in local_helper/tools/."
        continue
    }
    $found = Get-Command $tool -ErrorAction SilentlyContinue
    if ($found) {
        Write-Host "Found $tool on PATH at $($found.Source) -- copying it in."
        Copy-Item $found.Source $dest
    } else {
        Write-Warning "$tool was not found on PATH and is not in local_helper/tools/. The helper will fail to start until one is available (either on this machine's PATH, or placed manually into local_helper/tools/)."
    }
}

Write-Host "=== Step 3b: bundle the deno binary (Deno JS-runtime for yt-dlp signature challenges) ==="
# Discovered via real end-to-end testing of the frozen exe: the `deno`
# PyPI package (requirements-helper.txt) locates/downloads its actual
# deno.exe into the environment's own Scripts/ directory at pip-install
# time -- that lookup mechanism does not resolve inside a frozen
# PyInstaller app (no Scripts/ directory exists there), so yt-dlp fell
# back to no JS runtime at all ("No supported JavaScript runtime could be
# found"). Bundling the real binary into tools/ (already on PATH via
# launcher.py's _prepend_bundled_tools_to_path) lets yt-dlp's own PATH
# lookup find it, matching how cloudflared/ffmpeg are already handled.
$DenoDest = Join-Path $ToolsDir "deno.exe"
if (Test-Path $DenoDest) {
    Write-Host "deno.exe already present in local_helper/tools/."
} else {
    $BuildVenvDeno = Join-Path $BuildVenv "Scripts\deno.exe"
    if (Test-Path $BuildVenvDeno) {
        Write-Host "Found deno.exe in the build venv at $BuildVenvDeno -- copying it in."
        Copy-Item $BuildVenvDeno $DenoDest
    } else {
        $ExistingDeno = Get-Command deno.exe -ErrorAction SilentlyContinue
        if ($ExistingDeno) {
            Write-Host "Found deno.exe on PATH at $($ExistingDeno.Source) -- copying it in."
            Copy-Item $ExistingDeno.Source $DenoDest
        } else {
            Write-Warning "deno.exe was not found. YouTube extraction will still work for most videos, but formats requiring a JS signature challenge may be unavailable."
        }
    }
}

Write-Host "=== Step 4: PyInstaller build ==="
& $BuildPython -m PyInstaller `
    --name "Raza Audio URL Helper" `
    --onedir `
    --windowed `
    --noconfirm `
    --paths $RepoRoot `
    --paths (Join-Path $RepoRoot "src") `
    --hidden-import "audio_deepfake_detector.preprocessing.audio_loader" `
    --hidden-import "audio_deepfake_detector.utils.datatypes" `
    --hidden-import "app.analysis.video_url" `
    --hidden-import "app.analysis.media_ffmpeg" `
    --hidden-import "app.analysis.pot_provider" `
    --hidden-import "app.validation" `
    --hidden-import "app.errors" `
    --distpath (Join-Path $RepoRoot "dist") `
    (Join-Path $RepoRoot "local_helper\launcher.py")

Write-Host "=== Step 5: copy tools/ next to the built executable ==="
$DistTools = Join-Path $DistDir "tools"
New-Item -ItemType Directory -Force -Path $DistTools | Out-Null
Copy-Item (Join-Path $ToolsDir "*") $DistTools -Force -Recurse -ErrorAction SilentlyContinue

Write-Host "=== Step 6: smoke test (import only, no GUI/tunnel started) ==="
& $BuildPython -c "import local_helper.app, local_helper.api, local_helper.tunnel, local_helper.gui; print('local_helper package imports OK')"

Write-Host ""
Write-Host "Build complete: $DistDir\Raza Audio URL Helper.exe"
Write-Host "This script is not needed again unless the helper's code changes."
