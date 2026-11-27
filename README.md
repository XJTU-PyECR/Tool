# PyECR

PyECR is a desktop program for electrical conductivity relaxation analysis. It fits single and twofold relaxation, evaluates parameter sensitivity and conditional D-k error maps, and calculates distributions of characteristic times (DCT). The interface also provides forward simulation, batch fitting, comparison of independent fits, and data export.

## Requirements

- Windows 10 or 11, 64-bit.
- Python 3.14, 64-bit. Select the option to add Python to PATH during installation.
- An internet connection for the first launch, when the required packages are installed.

The package versions are listed in [requirements.txt](requirements.txt). A Python runtime is not included in this repository.

## Installation and startup

1. Download the repository as a ZIP file and extract it, or clone it with Git.
2. Keep `StartPyECR.cmd`, `PyECR.py`, `requirements.txt`, and `src` in the same folder.
3. Double-click `StartPyECR.cmd`. Wait for the dependency check to finish and the application window to open. Keep the console open while using PyECR.

The launcher creates an isolated environment at `%LOCALAPPDATA%\PyECR\V1.0\venv`. If `.venv\Scripts\python.exe` is already present in the source folder, it uses that environment instead. An existing environment must use 64-bit Python 3.14.

From PowerShell in the source folder, the equivalent command is:

```powershell
.\StartPyECR.cmd
```

To check the dependencies and build the interface without opening the main window:

```powershell
.\StartPyECR.cmd --check
```

The Python entry point is also available:

```powershell
py -3.14 PyECR.py
py -3.14 PyECR.py --check
```

If startup fails, retain the console output and check that Python 3.14 is installed and the source folder has been fully extracted.

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
