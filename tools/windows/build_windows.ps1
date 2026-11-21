# SPDX-License-Identifier: MPL-2.0
# Builds VisualDeck for Windows as one folder that runs without the development
# launcher, Git or an installed Python (PyInstaller), checks it with
# VisualDeck.exe --self-test, and zips it.
#
#   Run from the development folder:  powershell -ExecutionPolicy Bypass -File app\tools\windows\build_windows.ps1
#   (or double-click Build-VisualDeck.cmd next to Start-VisualDeck.cmd).
#
# Needs the development folder made by Start-VisualDeck.cmd (its Python 3.12 and app\).
# Uses its own Python environment in build\venv: the app's .venv is not touched.
# Writes: dist\VisualDeck\ (the program), dist\VisualDeck-<version>-windows.zip,
#         logs\build-<time>.log and logs\selftest-<time>.json.
# This is a test build for the owner's PC, not a release: see docs/BUILD.md.

$ErrorActionPreference = "Stop"
$App   = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Root  = Split-Path -Parent $App
$Work  = Join-Path $Root "build"
$Dist  = Join-Path $Root "dist"
$Logs  = Join-Path $Root "logs"
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
New-Item -ItemType Directory -Force -Path $Work, $Dist, $Logs | Out-Null
try { Start-Transcript -Path (Join-Path $Logs "build-$Stamp.log") | Out-Null } catch { }

function Step([string]$t) { Write-Host ""; Write-Host "== $t" -ForegroundColor Cyan }
function Fail([string]$t) {
    Write-Host "BUILD FAILED: $t" -ForegroundColor Red
    try { Stop-Transcript | Out-Null } catch { }
    exit 1
}

# The Python 3.12 that the app's environment was made from (pyvenv.cfg 'home').
Step "Python"
$cfg = Join-Path $App ".venv\pyvenv.cfg"
if (-not (Test-Path $cfg)) { Fail "No $cfg. Run Start-VisualDeck.cmd once first." }
$homeLine = Get-Content $cfg | Where-Object { $_ -match '^\s*home\s*=' } | Select-Object -First 1
$pyHome = ($homeLine -replace '^\s*home\s*=\s*', '').Trim()
$py = Join-Path $pyHome "python.exe"
if (-not (Test-Path $py)) { Fail "Python not found at $py (from $cfg)." }
& $py --version

Step "Build environment (build\venv)"
$venv = Join-Path $Work "venv"
$bpy = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $bpy)) {
    & $py -m venv $venv
    if ($LASTEXITCODE -ne 0) { Fail "could not create $venv" }
}
$ErrorActionPreference = "Continue"     # pip writes notices to stderr
& $bpy -m pip install --disable-pip-version-check -q -r (Join-Path $App "requirements.txt") "pyinstaller==6.11.1" "pefile==2023.2.7" 2>&1 |
    ForEach-Object { "$_" }
$pipRc = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($pipRc -ne 0) { Fail "pip install failed (exit $pipRc)" }

# Version: the code's git description, else the commit file the launcher keeps.
$version = "dev"
foreach ($g in @("git", "$env:ProgramFiles\Git\cmd\git.exe", "$env:LOCALAPPDATA\Programs\Git\cmd\git.exe")) {
    try {
        $v = (& $g -C $App describe --tags --always --dirty 2>$null)
        if ($LASTEXITCODE -eq 0 -and $v) { $version = "$v".Trim(); break }
    } catch { }
}
Write-Host "Version: $version"

Step "PyInstaller"
$ErrorActionPreference = "Continue"     # PyInstaller logs to stderr
& $bpy -m PyInstaller --noconfirm --clean --log-level WARN --distpath $Dist `
    --workpath (Join-Path $Work "pyinstaller") (Join-Path $PSScriptRoot "VisualDeck.spec") 2>&1 | ForEach-Object { "$_" }
$piRc = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($piRc -ne 0) { Fail "PyInstaller failed (exit $piRc)" }
$out = Join-Path $Dist "VisualDeck"
$exe = Join-Path $out "VisualDeck.exe"
if (-not (Test-Path $exe)) { Fail "no $exe" }

# The PyAV wheel's FFmpeg includes x264 and x265 (GPL). Put a build of the same FFmpeg
# major versions without GPL parts in its place,
# downloaded once and checked against its SHA-256.
Step "FFmpeg without GPL parts (LGPL build)"
$ffUrl = "https://github.com/FFmpeg-Builds/FFmpeg-Builds/releases/download/autobuild-2026-07-31-14-10/ffmpeg-n7.1.5-12-g1fdbca85aa-win64-lgpl-shared-7.1.zip"
$ffSha = "0f376f96fb38554ccefb1b2ae9c7c6a7b351f0e60a372b38262c320e8392c5d0"
$ffDir = Join-Path $Work "ffmpeg-lgpl"
$ffZip = Join-Path $ffDir (Split-Path -Leaf $ffUrl)
New-Item -ItemType Directory -Force -Path $ffDir | Out-Null
function ShaOf([string]$f) { if (Test-Path $f) { (Get-FileHash -Algorithm SHA256 -LiteralPath $f).Hash.ToLower() } else { "" } }
if ((ShaOf $ffZip) -ne $ffSha) {
    Write-Host "Downloading $ffUrl"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $oldProgress = $ProgressPreference; $ProgressPreference = "SilentlyContinue"
    try { Invoke-WebRequest -Uri $ffUrl -OutFile $ffZip -UseBasicParsing } catch { Fail "could not download the LGPL FFmpeg: $_" }
    $ProgressPreference = $oldProgress
}
if ((ShaOf $ffZip) -ne $ffSha) { Fail "the LGPL FFmpeg download does not match its SHA-256 ($ffZip)" }
$ffX = Join-Path $ffDir "unpacked"
if (Test-Path $ffX) { Remove-Item -LiteralPath $ffX -Recurse -Force }
Expand-Archive -LiteralPath $ffZip -DestinationPath $ffX
$ffCodec = Get-ChildItem -Path $ffX -Recurse -Filter "avcodec-61.dll" | Select-Object -First 1
if (-not $ffCodec) { Fail "no avcodec-61.dll in $ffZip" }
$ffLicense = Join-Path $ffCodec.Directory.Parent.FullName "LICENSE.txt"
& $bpy (Join-Path $PSScriptRoot "lgpl_ffmpeg.py") $out $ffCodec.DirectoryName --license $ffLicense --source $ffUrl
if ($LASTEXITCODE -ne 0) { Fail "could not put the LGPL FFmpeg into $out" }

foreach ($f in @("LICENSE", "README.md", "CHANGELOG.md", "THIRD_PARTY_NOTICES.md")) {
    $src = Join-Path $App $f
    if (Test-Path $src) { Copy-Item -LiteralPath $src -Destination $out -Force }
}
Set-Content -Path (Join-Path $out "VERSION.txt") -Encoding UTF8 -Value @(
    "VisualDeck $version (Windows test build, $(Get-Date -Format 'yyyy-MM-dd HH:mm'))",
    "Not a release: see CHANGELOG.md and docs/BUILD.md in the source.")

Step "Self-test of the built program"
$report = Join-Path $Logs "selftest-$Stamp.json"
$p = Start-Process -FilePath $exe -ArgumentList @("--self-test", "`"$report`"") -Wait -PassThru
if (-not (Test-Path $report)) { Fail "the self-test wrote no report (exit $($p.ExitCode))" }
$r = Get-Content $report -Raw | ConvertFrom-Json
foreach ($c in $r.checks.PSObject.Properties) {
    $v = $c.Value
    if ($v.ok) { Write-Host ("  ok    {0,-9} {1}" -f $c.Name, $v.detail) }
    else       { Write-Host ("  FAIL  {0,-9} {1}" -f $c.Name, $v.error) -ForegroundColor Red }
}
if (-not $r.passed) { Fail "self-test failed (report: $report)" }
if ("$($r.checks.ffmpeg.detail)" -notmatch "distribution is LGPL") {
    Fail "the built program's FFmpeg is not an LGPL build: $($r.checks.ffmpeg.detail)"
}

Step "Zip"
$safe = ($version -replace '[^A-Za-z0-9._-]', '_')
$zip = Join-Path $Dist "VisualDeck-$safe-windows.zip"
if (Test-Path $zip) { Remove-Item -LiteralPath $zip -Force }
Compress-Archive -Path $out -DestinationPath $zip
$mb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host "Built $exe"
Write-Host "Zip   $zip ($mb MB)"

Step "Installer (NSIS)"
$nsis = $null
foreach ($m in @("makensis", "${env:ProgramFiles(x86)}\NSIS\makensis.exe", "$env:ProgramFiles\NSIS\makensis.exe")) {
    $c = Get-Command $m -ErrorAction SilentlyContinue
    if ($c) { $nsis = $c.Source; break }
}
if ($nsis) {
    $setup = Join-Path $Dist "VisualDeck-$safe-setup.exe"
    & $nsis -V2 "/DVERSION=$safe" "/DDIST=$out" "/DOUT=$setup" (Join-Path $PSScriptRoot "installer.nsi")
    if ($LASTEXITCODE -ne 0) { Fail "makensis failed (exit $LASTEXITCODE)" }
    Write-Host "Setup $setup ($([math]::Round((Get-Item $setup).Length / 1MB, 1)) MB)"
} else {
    Write-Host "NSIS not found: no installer made (install NSIS 3, or run makensis on tools\windows\installer.nsi elsewhere)."
}
Write-Host "BUILD OK" -ForegroundColor Green
try { Stop-Transcript | Out-Null } catch { }
exit 0
