@echo off
REM SPDX-License-Identifier: MPL-2.0
rem VisualDeck development launcher. Double-click to start VisualDeck.
rem Options:  Start-VisualDeck.cmd test   (run the automated tests)
rem           Start-VisualDeck.cmd setup  (update and install only)
setlocal
title VisualDeck
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launcher\launcher.ps1" %*
set "rc=%ERRORLEVEL%"
if not "%rc%"=="0" (
  echo.
  echo The launcher stopped with an error ^(code %rc%^). The messages above and the logs folder say why.
  pause
)
exit /b %rc%
