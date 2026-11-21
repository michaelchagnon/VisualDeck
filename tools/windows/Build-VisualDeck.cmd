@echo off
REM SPDX-License-Identifier: MPL-2.0
rem Builds VisualDeck as a folder that runs without this launcher, Git or Python
rem (dist\VisualDeck\VisualDeck.exe and a zip). Put this file next to Start-VisualDeck.cmd.
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0app\tools\windows\build_windows.ps1"
set "rc=%ERRORLEVEL%"
echo.
if not "%rc%"=="0" (echo The build stopped with an error ^(code %rc%^). logs\build-*.log says why.) else (echo Done: dist\VisualDeck\VisualDeck.exe)
pause
exit /b %rc%
