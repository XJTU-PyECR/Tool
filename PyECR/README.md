# PyECR

PyECR is a desktop program for electrical conductivity relaxation analysis. It fits single and twofold relaxation, evaluates parameter sensitivity and conditional D-k error maps, and calculates distributions of characteristic times (DCT). The interface also provides forward simulation, batch fitting, comparison of independent fits, and data export.

## Requirements

- Windows 10 or 11, 64-bit.
- An internet connection for the first launch, to download Python if needed and install the required packages.

PyECR uses standard 64-bit Python 3.14. If a suitable interpreter is already available, the launcher uses it. Otherwise, the launcher offers to download the official CPython runtime from python.org. Python does not need to be installed beforehand. The package versions are listed in [requirements.txt](requirements.txt).

## Installation and startup

1. Download the repository as a ZIP file and extract it fully, or clone it with Git.
2. Keep `StartPyECR.cmd`, `SetupPyECR.ps1`, `PyECR.py`, `requirements.txt`, and `src` in the same folder.
3. Double-click `StartPyECR.cmd`. If Python 3.14 is missing, select `Y` when asked whether to download and set it up. Wait for the runtime and package installation to finish. Downloads may take several minutes.
4. The application opens when setup is complete. Keep the console open while using PyECR.

The optional runtime is downloaded from the Python Software Foundation's official release server, checked against the SHA-256 in the official release manifest, and placed at `%LOCALAPPDATA%\PyECR\Runtime\Python3146`. This does not require administrator rights or change the system PATH, registry, or other Python installations. The downloaded archive is cached at `%LOCALAPPDATA%\PyECR\Downloads`.

The launcher creates an isolated environment at `%LOCALAPPDATA%\PyECR\V1.0\venv`. A working `.venv\Scripts\python.exe` in the source folder takes priority. An unusable local `.venv` is ignored. If the cached PyECR environment is unusable, the launcher preserves it with an `.unusable-...` suffix and creates a replacement after locating a suitable Python runtime.

From PowerShell in the source folder, the equivalent command is:

```powershell
.\StartPyECR.cmd
```

To check dependencies and build the interface without opening the main window:

```powershell
.\StartPyECR.cmd --check
```

The check does not prompt to download Python. To explicitly allow runtime setup during the check:

```powershell
.\StartPyECR.cmd --install-runtime --check
```

If Python is already installed, the Python entry point is also available:

```powershell
py -3.14 PyECR.py
py -3.14 PyECR.py --check
```

## Startup troubleshooting

- **Python 3.14 is missing:** double-click `StartPyECR.cmd` and select `Y`. If automatic setup is unavailable, install the Windows installer (64-bit) from the [official Python download page](https://www.python.org/downloads/release/python-3146/), then run the launcher again. Installing the Python install manager alone may not install a Python runtime; with that manager, `py install 3.14` installs the required runtime.
- **Python download fails:** check access to `https://www.python.org` and retry, or install Python manually. Incomplete downloads are not executed.
- **Package installation fails:** check the pip error in the console and access to the Python package index. The launcher retries missing packages on the next run.
- **A ZIP was copied from another computer:** extract all source files before running. Virtual environments cannot be relied on after copying between computers; the launcher checks them before use.
- **Offline use:** complete the initial Python and package setup while connected to the internet. Subsequent launches can use the installed environment without downloading it again.

If startup still fails, retain the console output so that the failed step can be identified.

## Using the program

In Analysis, follow the left workflow: Import, Preprocess, Model & experiment, Run fit, Diagnostics, and Export. Confirm the signal type, units, sample dimensions, exposed faces, and gas-step conditions before fitting. Save projects and exported results in a separate working folder.

See the [User's Guide](PyECR_V1.0%20User%20Guide.pdf) for the mathematical models and detailed instructions, including D-k map colors, DCT, simulation, and export.

## Manuscript data

Manuscript case datasets are not included in this repository. The extracted input data and PyECR-exported numerical results will be supplied separately with the journal submission.

## License

Copyright belongs to Xi'an Jiaotong University. Use, modification, and redistribution are permitted for noncommercial purposes under the [LICENSE](LICENSE). Commercial use requires prior written permission from the copyright holder. Third-party packages retain their own licenses.

## Contact

Youjun Lu: yjlu@mail.xjtu.edu.cn  
Yihang Li: liyihang@xidian.edu.cn
