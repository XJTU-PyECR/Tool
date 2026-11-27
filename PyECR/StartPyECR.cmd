@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title PyECR V1.0
pushd "%~dp0" || exit /b 1
set "check_only="
set "install_runtime="

:arguments
if "%~1"=="" goto configure
if /i "%~1"=="--check" (
    set "check_only=1"
) else if /i "%~1"=="--install-runtime" (
    set "install_runtime=1"
) else (
    echo Unknown option: %~1
    popd
    exit /b 2
)
shift
goto arguments

:configure
set "PYTHON_MANAGER_AUTOMATIC_INSTALL=false"
set "PYLAUNCHER_ALLOW_INSTALL="
set "PYTHONIOENCODING=utf-8"
set "env_dir=%LOCALAPPDATA%\PyECR\V1.0\venv"
set "private_python=%LOCALAPPDATA%\PyECR\Runtime\Python3146\python.exe"
set "version_check=import struct, sys, sysconfig, venv, ensurepip; sys.exit(not (sys.version_info[:2] == (3, 14) and struct.calcsize('P') == 8 and not sysconfig.get_config_var('Py_GIL_DISABLED')))"
set "find_python=import sys; print(sys.executable)"
set "load_source=import sys; from pathlib import Path; sys.path.insert(0, str(Path('src').resolve()))"
if not exist "src\pyecr\gui\app.py" goto missing_source
if not exist "requirements.txt" goto missing_source
if not exist "SetupPyECR.ps1" goto missing_source

if not exist ".venv\Scripts\python.exe" goto cached_environment
".venv\Scripts\python.exe" -I -c "%version_check%" >nul 2>&1
if errorlevel 1 (
    echo The local .venv is not usable. PyECR will use its separate environment.
) else (
    set "env_dir=%CD%\.venv"
    goto prepare
)

:cached_environment
if not exist "%env_dir%\Scripts\python.exe" goto find_base
"%env_dir%\Scripts\python.exe" -I -c "%version_check%" >nul 2>&1
if not errorlevel 1 goto prepare
echo The previous PyECR environment is not usable and will be rebuilt.

:find_base
set "base_python="
call :try_python "%private_python%"
if defined base_python goto create_environment
for /f "delims=" %%P in ('py -3.14 -I -c "%find_python%" 2^>nul') do call :try_python "%%P"
if defined base_python goto create_environment
for /f "delims=" %%P in ('where.exe python.exe 2^>nul') do call :try_python "%%P"
if defined base_python goto create_environment
goto missing_python

:try_python
if defined base_python exit /b 0
if not exist "%~1" exit /b 0
if /i "%~dp1"=="%LOCALAPPDATA%\Microsoft\WindowsApps\" exit /b 0
"%~1" -I -c "%version_check%" >nul 2>&1
if not errorlevel 1 set "base_python=%~1"
exit /b 0

:missing_python
echo.
echo PyECR needs 64-bit Python 3.14. It is not installed or cannot be used.
echo PyECR can download a verified official Python runtime for this user.
echo This does not require administrator rights or change the system PATH.
echo An internet connection is required for Python and the analysis packages.
if defined install_runtime goto install_python
if defined check_only goto python_instructions
choice /c YN /n /m "Download and set up Python now? [Y/N]: "
if errorlevel 2 goto python_instructions
if errorlevel 1 goto install_python
goto python_instructions

:install_python
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File ".\SetupPyECR.ps1"
if errorlevel 1 goto python_instructions
call :try_python "%private_python%"
if not defined base_python goto python_instructions

:create_environment
if exist "%env_dir%" (
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File ".\SetupPyECR.ps1" -ResetEnvironment
    if errorlevel 1 goto failed
)
echo Creating the Python environment for PyECR.
"%base_python%" -I -m venv "%env_dir%"
if errorlevel 1 goto failed

:prepare
set "env_python=%env_dir%\Scripts\python.exe"
"%env_python%" -I -c "%version_check%" >nul 2>&1
if errorlevel 1 goto python_instructions
"%env_python%" -I -m pip --version >nul 2>&1
if errorlevel 1 (
    "%env_python%" -I -m ensurepip --upgrade
    if errorlevel 1 goto failed
)
echo Checking dependencies and installing any missing packages.
"%env_python%" -I -m pip --disable-pip-version-check --no-input install --quiet -r "requirements.txt"
if errorlevel 1 goto dependency_failed
if defined check_only goto check
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

:python_instructions
echo.
echo Python setup is not complete. Keep the files fully extracted, then try again.
echo To install Python manually, download Windows installer (64-bit) from:
echo https://www.python.org/downloads/release/python-3146/
echo After installing Python 3.14, double-click StartPyECR.cmd again.
goto failed

:dependency_failed
echo.
echo Python is ready, but the required analysis packages could not be installed.
echo Check your internet connection and the pip error above, then run again.
goto failed

:missing_source
echo Keep StartPyECR.cmd, SetupPyECR.ps1, requirements.txt, and src together.

:failed
echo PyECR could not start. Review the messages above, then try again.
if not defined check_only (
    echo Press any key to close this window.
    pause >nul
)
popd
exit /b 1

:done
popd
exit /b 0
