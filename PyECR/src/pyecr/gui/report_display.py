"""UI-only Reports destination validation. Never creates directories or files."""
from pathlib import Path
import re

VERSION = 'report-layout/1.0.0'
PARENT_MODE = 'Parent + new report name'
MANUAL_MODE = 'Full new directory (manual)'
CURRENT_SOURCE = 'Current analysis'
HISTORY_SOURCE = 'Historical captured snapshot'


def report_name(value: str) -> str:
    if not value or value != value.strip():
        raise ValueError('Enter one new report name without leading/trailing whitespace.')
    if value in ('.', '..') or any(ord(c) < 32 or c in '<>:"/\\|?*' for c in value) or value.endswith('.'):
        raise ValueError('Report name must be one folder name; path separators and Windows invalid characters are forbidden.')
    stem = value.split('.', 1)[0].upper()
    if stem in ('CON', 'PRN', 'AUX', 'NUL', 'CONIN$', 'CONOUT$') or re.fullmatch(r'(COM|LPT)[1-9¹²³]', stem):
        raise ValueError('Windows reserved folder name; choose another report name.')
    return value


def compose_target(parent: str, name: str) -> Path:
    if not str(parent).strip():
        raise ValueError('Choose or enter an existing parent directory.')
    p = Path(str(parent).strip()).expanduser().resolve()
    if not p.is_dir():
        raise ValueError('Parent directory must already exist.')
    return p / report_name(str(name))


def check_new_target(value: str | Path) -> Path:
    if not str(value).strip():
        raise ValueError('Enter a full new report directory.')
    p = Path(str(value).strip()).expanduser().resolve()
    report_name(p.name)
    if not p.parent.is_dir():
        raise ValueError('Parent directory must already exist.')
    if p.exists():
        raise FileExistsError('Target already exists; choose a new name. Existing reports are never overwritten.')
    return p
