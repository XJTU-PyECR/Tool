"""Start the PyECR desktop application on Windows."""

import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check the dependencies and build the interface without opening a window.",
    )
    parser.add_argument(
        "--install-runtime",
        action="store_true",
        help="Allow setup of the official Python runtime if Python 3.14 is missing.",
    )
    options = parser.parse_args()
    launcher = Path(__file__).resolve().with_name("StartPyECR.cmd")
    if not launcher.is_file():
        parser.exit(1, f"Launcher not found: {launcher}\n")

    command = [os.environ["ComSpec"], "/d", "/c", f".\\{launcher.name}"]
    if options.check:
        command.append("--check")
    if options.install_runtime:
        command.append("--install-runtime")
    return subprocess.run(command, cwd=launcher.parent).returncode


if __name__ == "__main__":
    raise SystemExit(main())
