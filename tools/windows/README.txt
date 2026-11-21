VisualDeck development folder
=============================

Start VisualDeck:  double-click Start-VisualDeck.cmd

The first start checks for Git and Python 3.12 (and offers to install them
with winget if missing), creates the app folder and installs Python packages.
That takes a few minutes. Later starts take seconds.

Getting updates
- Put a code bundle (git bundle create visualdeck.bundle main --tags) in
  updates\ and the next start picks it up.
- Or save VisualDeck-update.bundle into Downloads; the next start finds it
  there.
Each start lists the changes it applied.

Other options (run from a terminal in this folder)
  Start-VisualDeck.cmd test    run the automated tests (windows will flash
                               on screen for about a minute)
  Start-VisualDeck.cmd setup   update and install without starting the app

Folders
  app\       the code (a git checkout) and its Python environment
  updates\   the code bundle the launcher updates from
  logs\      launcher, package-install, app and test logs; system-info.txt

The launcher never deletes your files or overwrites edits you make in app\.
Your VisualDeck settings stay in %USERPROFILE%\.visualdeck as before.
