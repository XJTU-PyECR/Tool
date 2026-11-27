"""Existing-directory chooser; Windows Unicode shell API in a dedicated thread."""
import json
import subprocess
import sys
import os

def _choose(initial):
    import tkinter as tk
    from tkinter import filedialog
    root=tk.Tk();root.withdraw()
    try:
        return filedialog.askdirectory(parent=root, title='PyECR — Select existing report parent',
                                       initialdir=initial, mustexist=True)
    finally:root.destroy()

def choose_parent_directory(initial):
    if sys.platform=='win32':return _windows_choose(initial)
    completed=subprocess.run([sys.executable, str(__file__), str(initial)],
                             capture_output=True, text=True, encoding='utf-8',
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), check=True)
    return json.loads(completed.stdout)

def _windows_choose(initial):
    """Existing filesystem folders only; native Unicode shell API in the app's worker thread."""
    import ctypes
    from ctypes import wintypes
    user=ctypes.WinDLL('user32',use_last_error=True)
    shell=ctypes.WinDLL('shell32',use_last_error=True)
    ole=ctypes.WinDLL('ole32')
    enum_callback=ctypes.WINFUNCTYPE(wintypes.BOOL,wintypes.HWND,wintypes.LPARAM)
    handles=[]
    user.GetWindowThreadProcessId.argtypes=(wintypes.HWND,ctypes.POINTER(wintypes.DWORD))
    user.IsWindowVisible.argtypes=(wintypes.HWND,)
    @enum_callback
    def visit(hwnd,param):
        pid=wintypes.DWORD();user.GetWindowThreadProcessId(hwnd,ctypes.byref(pid))
        if pid.value==os.getpid() and user.IsWindowVisible(hwnd):handles.append(hwnd)
        return True
    user.EnumWindows.argtypes=(enum_callback,wintypes.LPARAM);user.EnumWindows(visit,0)
    callback_type=ctypes.WINFUNCTYPE(ctypes.c_int,wintypes.HWND,wintypes.UINT,wintypes.LPARAM,wintypes.LPARAM)
    initial_text=ctypes.create_unicode_buffer(str(initial))
    user.SendMessageW.argtypes=(wintypes.HWND,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM)
    user.SendMessageW.restype=wintypes.LPARAM
    @callback_type
    def initialize(hwnd,message,param,data):
        if message==1:user.SendMessageW(hwnd,0x467,1,ctypes.cast(initial_text,ctypes.c_void_p).value)
        return 0
    class BrowseInfo(ctypes.Structure):
        _fields_=[('owner',wintypes.HWND),('root',ctypes.c_void_p),('display',wintypes.LPWSTR),('title',wintypes.LPCWSTR),
                  ('flags',wintypes.UINT),('callback',callback_type),('data',wintypes.LPARAM),('image',ctypes.c_int)]
    display=ctypes.create_unicode_buffer(32768)
    info=BrowseInfo(handles[0] if handles else None,None,ctypes.cast(display,wintypes.LPWSTR),'PyECR — Select an existing report parent; Cancel keeps the current location.',0x251,initialize,0,0)
    shell.SHBrowseForFolderW.argtypes=(ctypes.POINTER(BrowseInfo),);shell.SHBrowseForFolderW.restype=ctypes.c_void_p
    shell.SHGetPathFromIDListEx.argtypes=(ctypes.c_void_p,wintypes.LPWSTR,wintypes.DWORD,wintypes.DWORD);shell.SHGetPathFromIDListEx.restype=wintypes.BOOL
    ole.CoInitializeEx.argtypes=(ctypes.c_void_p,wintypes.DWORD);ole.CoInitializeEx.restype=ctypes.c_long
    ole.CoTaskMemFree.argtypes=(ctypes.c_void_p,)
    initialized=ole.CoInitializeEx(None,2)
    if initialized<0:raise OSError('Native directory chooser COM initialization failed')
    item=None
    try:
        item=shell.SHBrowseForFolderW(ctypes.byref(info))
        if not item:return ''
        result=ctypes.create_unicode_buffer(32768)
        if not shell.SHGetPathFromIDListEx(item,result,len(result),0):raise ValueError('Select an existing filesystem directory')
        return result.value
    finally:
        if item:ole.CoTaskMemFree(item)
        ole.CoUninitialize()

if __name__=='__main__':
    print(json.dumps(_choose(sys.argv[1]), ensure_ascii=True))
