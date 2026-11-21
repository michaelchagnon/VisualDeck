# SPDX-License-Identifier: MPL-2.0
# VisualDeck development launcher for Windows (Windows PowerShell 5.1 or PowerShell 7).
#
# Started by Start-VisualDeck.cmd, which sits one folder above this script:
#
#   VisualDeck-dev\
#     Start-VisualDeck.cmd        <- the one thing you run
#     launcher\launcher.ps1       <- this script
#     updates\visualdeck.bundle   <- the code, as a git bundle (replaced by updates)
#     app\                        <- git checkout + .venv, created on first run
#     logs\                       <- launcher, pip, app and test logs
#
# Every start: pick up a newer VisualDeck-update*.bundle from Downloads (if any),
# fast-forward app\ to the newest code, list what changed, install Python
# packages only when the requirements changed, then start VisualDeck.
#
# Modes:  run (default) | test (run the automated tests) | setup (no app start)
# It never deletes your files, never force-overwrites local edits in app\, and
# installs Python/Git only after asking.

param([string]$Mode = "run")

$ErrorActionPreference = "Stop"
$Mode = $Mode.ToLowerInvariant()
if (@("run", "test", "setup") -notcontains $Mode) {
    Write-Host "Unknown option '$Mode'. Use: Start-VisualDeck.cmd [run|test|setup]" -ForegroundColor Red
    exit 2
}

$Root    = Split-Path -Parent $PSScriptRoot
$App     = Join-Path $Root "app"
$Updates = Join-Path $Root "updates"
$Logs    = Join-Path $Root "logs"
$Bundle  = Join-Path $Updates "visualdeck.bundle"
$Stamp   = Get-Date -Format "yyyyMMdd-HHmmss"
New-Item -ItemType Directory -Force -Path $Updates, $Logs | Out-Null
try { Start-Transcript -Path (Join-Path $Logs "launcher-$Stamp.log") | Out-Null } catch { }

$script:StepNo = 0
$TotalSteps = 6

function Step([string]$Text) {
    $script:StepNo++
    Write-Host ""
    Write-Host ("[{0}/{1}] {2}" -f $script:StepNo, $TotalSteps, $Text) -ForegroundColor Cyan
}

function Info([string]$Text) { Write-Host "      $Text" }
function Warn([string]$Text) { Write-Host "      $Text" -ForegroundColor Yellow }

function Fail([string]$Text) {
    Write-Host ""
    Write-Host "ERROR: $Text" -ForegroundColor Red
    Write-Host "Logs are in: $Logs"
    try { Stop-Transcript | Out-Null } catch { }
    exit 1
}

function Ask-YesNo([string]$Question) {
    $answer = Read-Host "      $Question [Y/n]"
    return ($answer -eq "" -or $answer -match "^(y|yes)$")
}

# Run a native command and return its trimmed stdout, or $null if it fails.
# Errors and stderr are swallowed: this is only for probing.
function Probe([string]$Exe, [string[]]$Arguments) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $out = & $Exe @Arguments 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        if ($null -eq $out) { return "" }
        return (($out | Out-String).Trim())
    } catch {
        return $null
    } finally {
        $ErrorActionPreference = $old
    }
}

function Has-Winget { return [bool](Get-Command winget -ErrorAction SilentlyContinue) }

function Install-WithWinget([string]$Id, [string]$Name, [bool]$TryUserScope) {
    if (-not (Has-Winget)) {
        Warn "winget is not available on this PC, so $Name cannot be installed automatically."
        return $false
    }
    if (-not (Ask-YesNo "Install $Name now with winget?")) { return $false }
    $common = @("install", "--id", $Id, "-e", "--accept-package-agreements", "--accept-source-agreements")
    if ($TryUserScope) {
        & winget @($common + @("--scope", "user"))
        if ($LASTEXITCODE -eq 0) { return $true }
        Warn "Per-user install was not available; trying the standard installer (Windows may ask for permission)."
    }
    & winget @common
    return ($LASTEXITCODE -eq 0)
}

# ---------------------------------------------------------------- git

function Find-Git {
    $cmd = Get-Command git -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in @("$env:ProgramFiles\Git\cmd\git.exe", "$env:LOCALAPPDATA\Programs\Git\cmd\git.exe")) {
        if ($p -and (Test-Path $p)) { return $p }
    }
    return $null
}

# ---------------------------------------------------------------- Python 3.12

# VisualDeck pins packages (e.g. pywin32 306, NumPy 2.1.2) that publish
# Windows wheels for Python 3.12; other versions may fail to install.
function Find-Python312 {
    # Each probe passes -c, so no candidate can ever open an interactive prompt.
    $check = @("-c", "import sys; print(sys.executable if sys.version_info[:2] == (3, 12) else '')")
    $candidates = @(
        @{ Exe = "py";                                                Args = (@("-3.12") + $check) },
        @{ Exe = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"; Args = $check },
        @{ Exe = "$env:ProgramFiles\Python312\python.exe";            Args = $check },
        @{ Exe = "python3.12";                                        Args = $check },
        @{ Exe = "python";                                            Args = $check }
    )
    foreach ($c in $candidates) {
        $exe = $c.Exe
        if ($exe -like "*\*" -and -not (Test-Path $exe)) { continue }
        if ($exe -notlike "*\*" -and -not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
        $found = Probe $exe $c.Args
        if ($found -and (Test-Path $found)) { return $found }
    }
    return $null
}

function Venv-Python {
    foreach ($p in @((Join-Path $App ".venv\Scripts\python.exe"), (Join-Path $App ".venv/bin/python"))) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

# ---------------------------------------------------------------- updates

function Get-DownloadsFolder {
    try {
        $p = (New-Object -ComObject Shell.Application).NameSpace("shell:Downloads").Self.Path
        if ($p -and (Test-Path $p)) { return $p }
    } catch { }
    $p = Join-Path $env:USERPROFILE "Downloads"
    if (Test-Path $p) { return $p }
    return $null
}

# A VisualDeck-update*.bundle saved from the chat lands in Downloads; take the
# newest one if it is newer than the bundle we already have.
function Import-DownloadedBundle([string]$Git) {
    $dl = Get-DownloadsFolder
    if (-not $dl) { return }
    $candidates = @(Get-ChildItem -Path $dl -Filter "VisualDeck-update*.bundle" -File -ErrorAction SilentlyContinue |
                    Sort-Object LastWriteTime -Descending)
    if ($candidates.Count -eq 0) { return }
    $newest = $candidates[0]
    if ((Test-Path $Bundle) -and ($newest.LastWriteTime -le (Get-Item $Bundle).LastWriteTime)) { return }
    if (Test-Path (Join-Path $App ".git")) {
        $ok = Probe $Git @("-C", $App, "bundle", "verify", "--quiet", $newest.FullName)
        if ($null -eq $ok) {
            Warn "Ignoring $($newest.Name) in Downloads: git says it is not a valid VisualDeck bundle."
            return
        }
    }
    Copy-Item -LiteralPath $newest.FullName -Destination $Bundle -Force
    Info "Took the newer update from Downloads: $($newest.Name)"
}

function Update-App([string]$Git) {
    if (-not (Test-Path $Bundle)) {
        if (Test-Path (Join-Path $App ".git")) {
            Warn "No update bundle found; using the code already in $App"
            return
        }
        Fail "No code bundle found at $Bundle. Save VisualDeck-update.bundle from the chat into your Downloads folder and run this again."
    }

    if (-not (Test-Path (Join-Path $App ".git"))) {
        if (Test-Path $App) {
            if (@(Get-ChildItem -Force -LiteralPath $App).Count -gt 0) {
                Fail "$App exists but is not a VisualDeck checkout. Move it aside and run this again."
            }
        }
        Info "Creating the app folder from the bundle"
        & $Git clone --branch main $Bundle $App
        if ($LASTEXITCODE -ne 0) { Fail "git clone from $Bundle failed." }
        return
    }

    $before = Probe $Git @("-C", $App, "rev-parse", "HEAD")
    & $Git -C $App fetch --quiet --tags $Bundle "+refs/heads/main:refs/remotes/updates/main"
    if ($LASTEXITCODE -ne 0) {
        Warn "Could not read the update bundle; starting the code already installed."
        return
    }
    $latest = Probe $Git @("-C", $App, "rev-parse", "refs/remotes/updates/main")
    if ($before -eq $latest) {
        Info "Already up to date."
        return
    }
    $isBehind = Probe $Git @("-C", $App, "merge-base", "--is-ancestor", "HEAD", "refs/remotes/updates/main")
    if ($null -eq $isBehind) {
        $isAhead = Probe $Git @("-C", $App, "merge-base", "--is-ancestor", "refs/remotes/updates/main", "HEAD")
        if ($null -ne $isAhead) {
            Info "The installed code is newer than the bundle; nothing to update."
        } else {
            Warn "The update cannot be applied automatically (your app folder and the update have diverged)."
            Warn "Starting the code already installed. The local changes need merging by hand."
        }
        return
    }
    $dirty = Probe $Git @("-C", $App, "status", "--porcelain", "--untracked-files=no")
    if ($dirty) {
        Warn "You have edited files in $App; trying to update without touching them."
    }
    & $Git -C $App merge --ff-only --quiet "refs/remotes/updates/main"
    if ($LASTEXITCODE -ne 0) {
        Warn "Update skipped: it would overwrite your local edits. Starting the code already installed."
        return
    }
    Write-Host "      Updated. New changes:" -ForegroundColor Green
    & $Git -C $App --no-pager log --no-decorate --format="        %h  %s" "$before..HEAD"
}

function Update-LauncherCopy {
    # The repository carries the launcher too; refresh our copy for next start.
    $repoCopy = Join-Path $App "tools\windows\launcher.ps1"
    $mine = Join-Path $PSScriptRoot "launcher.ps1"
    if (-not (Test-Path $repoCopy)) { return }
    if ((Get-FileHash $repoCopy).Hash -ne (Get-FileHash $mine).Hash) {
        Copy-Item -LiteralPath $repoCopy -Destination $mine -Force
        Info "The launcher itself was updated; the new version takes effect next start."
    }
}

# ---------------------------------------------------------------- Python packages

function Install-Requirements([string]$VenvPy) {
    $files = @("requirements.txt")
    if ($Mode -eq "test") { $files += "requirements-dev.txt" }
    $hashText = ($files | ForEach-Object { (Get-FileHash (Join-Path $App $_) -Algorithm SHA256).Hash }) -join ","
    $marker = Join-Path $App ".venv\visualdeck-$Mode-requirements.sha256"
    if ((Test-Path $marker) -and ((Get-Content $marker -Raw).Trim() -eq $hashText)) {
        Info "Python packages are up to date."
        return
    }
    Info "Installing Python packages (first run downloads roughly 150-250 MB; later runs skip this)."
    & $VenvPy -m pip install --disable-pip-version-check --log (Join-Path $Logs "pip-$Stamp.log") -r (Join-Path $App $files[-1])
    if ($LASTEXITCODE -ne 0) { Fail "Package installation failed. Details: $(Join-Path $Logs "pip-$Stamp.log")" }
    Set-Content -Path $marker -Value $hashText
}

function Write-SystemInfo([string]$VenvPy, [string]$Git) {
    $lines = @("VisualDeck launcher system info  $Stamp")
    try {
        $os = Get-CimInstance Win32_OperatingSystem
        $lines += "OS: $($os.Caption) $($os.Version) $($os.OSArchitecture)"
        foreach ($g in @(Get-CimInstance Win32_VideoController)) {
            $mode = "no display attached"
            if ($g.CurrentHorizontalResolution) {
                $mode = "$($g.CurrentHorizontalResolution)x$($g.CurrentVerticalResolution) @ $($g.CurrentRefreshRate) Hz"
            }
            $lines += "GPU: $($g.Name) | driver $($g.DriverVersion) | $mode"
        }
        $cpu = @(Get-CimInstance Win32_Processor)[0]
        $lines += "CPU: $($cpu.Name) ($($cpu.NumberOfLogicalProcessors) threads)"
    } catch {
        $lines += "OS/GPU details unavailable: $($_.Exception.Message)"
    }
    try {
        $tz = Get-TimeZone
        $lines += "Time zone (Windows): $($tz.Id), UTC offset $($tz.BaseUtcOffset); TZ variable: '$env:TZ'"
    } catch {
        $lines += "Time zone (Windows): unavailable"
    }
    $lines += "Python: " + (Probe $VenvPy @("-c", "import sys; print(sys.version)"))
    $lines += "Python local time: " + (Probe $VenvPy @("-c", "import datetime, time; print(datetime.datetime.now().astimezone().isoformat(timespec='seconds'), time.tzname)"))
    $lines += "Git: " + (Probe $Git @("--version"))
    $lines += "Code: " + (Probe $Git @("-C", $App, "log", "-1", "--format=%h %cd %s", "--date=short"))
    Set-Content -Path (Join-Path $Logs "system-info.txt") -Value $lines
}

function Prune-Logs {
    foreach ($pattern in @("launcher-*.log", "pip-*.log", "app-*.log", "tests-*.txt")) {
        Get-ChildItem -Path $Logs -Filter $pattern -File -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -Skip 20 |
            Remove-Item -Force -ErrorAction SilentlyContinue
    }
}

# ================================================================== main

Write-Host "VisualDeck development launcher ($Mode)" -ForegroundColor White
Write-Host "Folder: $Root"
# Clear the "downloaded from the internet" mark so Windows stops asking each time.
try {
    Get-ChildItem -Path (Join-Path $Root "*.cmd"), (Join-Path $PSScriptRoot "*.ps1") -File -ErrorAction SilentlyContinue |
        Unblock-File
} catch { }

Step "Checking for Git"
$git = Find-Git
if (-not $git) {
    Warn "Git is needed to apply updates and was not found."
    if (Install-WithWinget "Git.Git" "Git" $false) { $git = Find-Git }
    if (-not $git) { Fail "Git is not installed. Install it from https://git-scm.com/download/win and run this again." }
}
Info (Probe $git @("--version"))

Step "Getting the latest VisualDeck code"
Import-DownloadedBundle $git
Update-App $git
Update-LauncherCopy
Info ("Version: " + (Probe $git @("-C", $App, "log", "-1", "--format=%h  %s  (%cr)")))

Step "Checking for Python 3.12"
$venvPy = Venv-Python
if (-not $venvPy) {
    $py = Find-Python312
    if (-not $py) {
        Warn "Python 3.12 was not found (VisualDeck's pinned packages need exactly 3.12 on Windows)."
        if (Install-WithWinget "Python.Python.3.12" "Python 3.12" $true) { $py = Find-Python312 }
        if (-not $py) { Fail "Python 3.12 is not installed. Get it from https://www.python.org/downloads/windows/ and run this again." }
    }
    Info "Using $py"
    Info "Creating the app's private Python environment (app\.venv)"
    & $py -m venv (Join-Path $App ".venv")
    if ($LASTEXITCODE -ne 0) { Fail "Could not create the Python environment." }
    $venvPy = Venv-Python
    if (-not $venvPy) { Fail "The Python environment was created but its python.exe is missing." }
} else {
    Info ("Using the app's Python environment: " + (Probe $venvPy @("--version")))
}

Step "Checking Python packages"
Install-Requirements $venvPy
Write-SystemInfo $venvPy $git
Prune-Logs

# A TZ variable in Linux form (e.g. "America/New_York") is not understood by
# the Windows C runtime, which then treats local time as UTC. Start Python
# without it (this process only; the system setting is not changed).
if ($env:TZ -and $env:TZ.Contains("/")) {
    Info "Ignoring TZ=$($env:TZ) for VisualDeck (Windows reads the time zone from its own settings)"
    Remove-Item Env:TZ
}

$appLog = Join-Path $env:USERPROFILE ".visualdeck\app.log"
$runStarted = Get-Date
$rc = 0
Push-Location $App
try {
    if ($Mode -eq "test") {
        Step "Running the automated tests"
        Warn "Test windows will flash on your main display for about a minute. Please don't click them."
        $results = Join-Path $Logs "tests-$Stamp.txt"
        $old = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $venvPy -m pytest -p no:cacheprovider -rfEs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $results
        $rc = $LASTEXITCODE
        $ErrorActionPreference = $old
        Info "Test results saved to $results"
    } elseif ($Mode -eq "run") {
        Step "Starting VisualDeck (close its window to finish)"
        & $venvPy (Join-Path $App "src\app.py")
        $rc = $LASTEXITCODE
    } else {
        Step "Setup complete (setup mode does not start the app)"
    }
} finally {
    Pop-Location
}

$script:StepNo = $TotalSteps - 1
Step "Saving logs"
# Copy the app log only if VisualDeck wrote to it during this run (test and
# setup modes do not start the app, so the log would be an old copy).
if ((Test-Path $appLog) -and ((Get-Item -LiteralPath $appLog).LastWriteTime -ge $runStarted)) {
    Copy-Item -LiteralPath $appLog -Destination (Join-Path $Logs "app-$Stamp.log") -Force
    Info "App log copied to logs\app-$Stamp.log"
} else {
    Info "VisualDeck did not run in this session, so no app log was copied."
}
if ($rc -ne 0) { Warn "Finished with exit code $rc." }
try { Stop-Transcript | Out-Null } catch { }
exit $rc
