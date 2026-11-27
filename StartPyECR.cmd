@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title PyECR V1.0
pushd "%~dp0" || exit /b 1

set "env_dir=%LOCALAPPDATA%\PyECR\V1.0\venv"
if exist ".venv\Scripts\python.exe" set "env_dir=%CD%\.venv"
set "env_python=%env_dir%\Scripts\python.exe"
set "version_check=import struct, sys; sys.exit(not (sys.version_info[:2] == (3, 14) and struct.calcsize('P') == 8))"
set "load_source=import sys; from pathlib import Path; sys.path.insert(0, str(Path('src').resolve()))"

if not exist "src\pyecr\gui\app.py" goto missing_source
if not exist "requirements.txt" goto missing_source
if exist "%env_python%" goto prepare

echo Creating the Python environment for the first launch of PyECR.
py -3.14 -I -c "%version_check%" >nul 2>&1
if not errorlevel 1 (
    py -3.14 -I -m venv "%env_dir%"
    if errorlevel 1 goto failed
    goto prepare
)

set "base_python="
for /f "delims=" %%P in ('where.exe python.exe 2^>nul') do (
    if not defined base_python (
        "%%P" -I -c "%version_check%" >nul 2>&1
        if not errorlevel 1 set "base_python=%%P"
    )
)
if not defined base_python goto missing_python
"%base_python%" -I -m venv "%env_dir%"
if errorlevel 1 goto failed

:prepare
"%env_python%" -I -c "%version_check%"
if errorlevel 1 (
    echo This environment requires 64-bit Python 3.14. Check which interpreter it uses.
    goto failed
)
"%env_python%" -I -m pip --version >nul 2>&1
if errorlevel 1 (
    "%env_python%" -I -m ensurepip --upgrade
    if errorlevel 1 goto failed
)
echo Checking dependencies and installing any missing packages.
"%env_python%" -I -m pip --disable-pip-version-check --no-input install --quiet -r "requirements.txt"
if errorlevel 1 goto failed
if /i "%~1"=="--check" goto check

echo Starting PyECR V1.0. Keep this window open until you close the application.
"%env_python%" -I -B -c "%load_source%; from pyecr.gui import main; main()"
if errorlevel 1 goto failed
goto done

:check
"%env_python%" -I -m pip check
if errorlevel 1 goto failed
"%env_python%" -I -B -c "%load_source%; import numpy, scipy, pandas, matplotlib, openpyxl, xlrd, PIL; import dearpygui.dearpygui as dpg; from pyecr.gui import PyecrGuiApp; app = PyecrGuiApp(); app.build(); app.controller.shutdown(wait=True); dpg.destroy_context(); print('PyECR V1.0 GUI check passed.')"
if errorlevel 1 goto failed
goto done

:missing_python
echo Could not find 64-bit Python 3.14. Install the 64-bit version for Windows, select the option to add Python to PATH, then double-click StartPyECR.cmd again.
goto failed

:missing_source
echo Place StartPyECR.cmd in the same folder as the src directory and requirements.txt.
:failed
echo PyECR could not start. Review the messages above, then try again.
if /i not "%~1"=="--check" (
    echo Press any key to close this window.
    pause >nul
)
popd
exit /b 1

:done
popd
exit /b 0
