"""Unicode Windows data-file picker. No parsing, imports, or DPG calls.

Call from a dedicated thread: DearPyGui's Windows directory iterator can abort
the process on non-ASCII paths. Use only wide-character Win32 APIs, retaining
the exact filename; no short-name, encoding, or temporary-file workaround.
"""
from __future__ import annotations

import os
import sys

TITLE = "PyECR — Select CSV, TXT, or XLSX"
FILTER = (
    "ECR data (*.csv;*.txt;*.xlsx)\0*.csv;*.txt;*.xlsx\0"
    "CSV (*.csv)\0*.csv\0Text (*.txt)\0*.txt\0"
    "Excel (*.xlsx)\0*.xlsx\0All files (*.*)\0*.*\0\0"
)


def choose_data_file(initial_directory: str) -> str:
    """Return an existing full Unicode path, or '' for a genuine cancel."""
    if sys.platform != "win32":
        raise NotImplementedError("Windows data picker is unavailable on this platform")
    return _windows_choose(initial_directory)


def _windows_choose(initial_directory: str) -> str:
    import ctypes
    from ctypes import wintypes

    user = ctypes.WinDLL("user32", use_last_error=True)
    common = ctypes.WinDLL("comdlg32", use_last_error=True)
    ole = ctypes.WinDLL("ole32")
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    handles = []
    user.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    user.IsWindowVisible.argtypes = (wintypes.HWND,)

    @callback_type
    def visit(hwnd, _):
        pid = wintypes.DWORD()
        user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == os.getpid() and user.IsWindowVisible(hwnd):
            handles.append(hwnd)
        return True

    user.EnumWindows.argtypes = (callback_type, wintypes.LPARAM)
    user.EnumWindows(visit, 0)

    class OpenFileNameW(ctypes.Structure):
        _fields_ = [
            ("lStructSize", wintypes.DWORD), ("hwndOwner", wintypes.HWND),
            ("hInstance", wintypes.HINSTANCE), ("lpstrFilter", wintypes.LPCWSTR),
            ("lpstrCustomFilter", wintypes.LPWSTR), ("nMaxCustFilter", wintypes.DWORD),
            ("nFilterIndex", wintypes.DWORD), ("lpstrFile", wintypes.LPWSTR),
            ("nMaxFile", wintypes.DWORD), ("lpstrFileTitle", wintypes.LPWSTR),
            ("nMaxFileTitle", wintypes.DWORD), ("lpstrInitialDir", wintypes.LPCWSTR),
            ("lpstrTitle", wintypes.LPCWSTR), ("Flags", wintypes.DWORD),
            ("nFileOffset", wintypes.WORD), ("nFileExtension", wintypes.WORD),
            ("lpstrDefExt", wintypes.LPCWSTR), ("lCustData", wintypes.LPARAM),
            ("lpfnHook", ctypes.c_void_p), ("lpTemplateName", wintypes.LPCWSTR),
            ("pvReserved", ctypes.c_void_p), ("dwReserved", wintypes.DWORD),
            ("FlagsEx", wintypes.DWORD),
        ]

    buffer = ctypes.create_unicode_buffer(32768)
    request = OpenFileNameW()
    request.lStructSize = ctypes.sizeof(request)
    request.hwndOwner = handles[0] if handles else None  # Only this process.
    request.lpstrFilter = FILTER
    request.nFilterIndex = 1
    request.lpstrFile = ctypes.cast(buffer, wintypes.LPWSTR)
    request.nMaxFile = len(buffer)
    request.lpstrInitialDir = str(initial_directory)
    request.lpstrTitle = TITLE
    # EXPLORER | FILEMUSTEXIST | PATHMUSTEXIST | NOCHANGEDIR |
    # DONTADDTORECENT | HIDEREADONLY. No multi-select or silent file creation.
    request.Flags = 0x80000 | 0x1000 | 0x800 | 0x8 | 0x2000000 | 0x4
    common.GetOpenFileNameW.argtypes = (ctypes.POINTER(OpenFileNameW),)
    common.GetOpenFileNameW.restype = wintypes.BOOL
    common.CommDlgExtendedError.restype = wintypes.DWORD
    ole.CoInitializeEx.argtypes = (ctypes.c_void_p, wintypes.DWORD)
    ole.CoInitializeEx.restype = ctypes.c_long
    initialized = ole.CoInitializeEx(None, 2)
    if initialized < 0:
        raise OSError("Native file picker COM initialization failed")
    try:
        if common.GetOpenFileNameW(ctypes.byref(request)):
            return buffer.value
        error = common.CommDlgExtendedError()
        if error:
            raise OSError(f"Native file picker failed (CommDlgExtendedError=0x{error:04x})")
        return ""
    finally:
        ole.CoUninitialize()
