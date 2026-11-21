; SPDX-License-Identifier: MPL-2.0
; VisualDeck installer (NSIS 3). Installs the folder that build_windows.ps1 makes
; (dist\VisualDeck) for the current user, without administrator rights, with a
; Start menu entry, a desktop shortcut and an uninstaller, and offers to start
; VisualDeck at the end. Nothing else needs installing: FFmpeg (the LGPL build),
; Python and every library are inside the folder.
;
;   makensis /DVERSION=1.0.0 /DDIST=..\..\..\dist\VisualDeck /DOUT=..\..\..\dist\VisualDeck-1.0.0-setup.exe installer.nsi
;
; makensis also runs on Linux (the nsis package), so the installer can be made
; from a copy of dist\VisualDeck there.

Unicode true
SetCompressor /SOLID lzma
RequestExecutionLevel user          ; per-user: no administrator prompt

!ifndef VERSION
  !define VERSION "dev"
!endif
!ifndef DIST
  !define DIST "..\..\..\dist\VisualDeck"
!endif
!ifndef OUT
  !define OUT "VisualDeck-${VERSION}-setup.exe"
!endif
!define NAME "VisualDeck"
!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\VisualDeck"

Name "${NAME} ${VERSION}"
OutFile "${OUT}"
InstallDir "$LOCALAPPDATA\Programs\VisualDeck"
InstallDirRegKey HKCU "Software\VisualDeck" "InstallDir"
Icon "${DIST}\_internal\src\visualdeck.ico"
UninstallIcon "${DIST}\_internal\src\visualdeck.ico"
BrandingText "${NAME} ${VERSION}"

!include "MUI2.nsh"
!define MUI_ICON "${DIST}\_internal\src\visualdeck.ico"
!define MUI_UNICON "${DIST}\_internal\src\visualdeck.ico"
!define MUI_WELCOMEPAGE_TEXT "This installs ${NAME} ${VERSION} for you (no administrator rights needed).$\r$\n$\r$\nVisualDeck is free software under the Mozilla Public License 2.0; the libraries it includes and their licenses are listed in THIRD_PARTY_NOTICES.md in the installation folder."
!define MUI_FINISHPAGE_RUN "$INSTDIR\VisualDeck.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Start VisualDeck now"
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Section "VisualDeck" SecMain
  SetShellVarContext current
  ; An update replaces the program's own files only (never the user's shows or
  ; ~\.visualdeck); old libraries must not stay behind next to the new ones.
  IfFileExists "$INSTDIR\VisualDeck.exe" 0 +2
    RMDir /r "$INSTDIR\_internal"
  SetOutPath "$INSTDIR"
  File /r "${DIST}\*.*"
  WriteUninstaller "$INSTDIR\Uninstall.exe"

  CreateDirectory "$SMPROGRAMS\VisualDeck"
  CreateShortcut "$SMPROGRAMS\VisualDeck\VisualDeck.lnk" "$INSTDIR\VisualDeck.exe"
  CreateShortcut "$SMPROGRAMS\VisualDeck\Uninstall VisualDeck.lnk" "$INSTDIR\Uninstall.exe"
  CreateShortcut "$DESKTOP\VisualDeck.lnk" "$INSTDIR\VisualDeck.exe"

  WriteRegStr HKCU "Software\VisualDeck" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayName" "${NAME}"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNINST_KEY}" "Publisher" "The VisualDeck Project"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayIcon" "$INSTDIR\VisualDeck.exe"
  WriteRegStr HKCU "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINST_KEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKCU "${UNINST_KEY}" "QuietUninstallString" '"$INSTDIR\Uninstall.exe" /S'
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoRepair" 1
SectionEnd

Section "Uninstall"
  SetShellVarContext current
  Delete "$DESKTOP\VisualDeck.lnk"
  RMDir /r "$SMPROGRAMS\VisualDeck"
  ; Only VisualDeck's own folder; shows and settings (~\.visualdeck) stay.
  IfFileExists "$INSTDIR\VisualDeck.exe" 0 +2
    RMDir /r "$INSTDIR"
  DeleteRegKey HKCU "${UNINST_KEY}"
  DeleteRegKey HKCU "Software\VisualDeck"
SectionEnd
