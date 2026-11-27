"""Exclusive sibling staging with the destination parent's Windows permissions."""
import os
from pathlib import Path
import secrets
import tempfile


def create_export_staging(parent, *, prefix):
    parent=Path(parent)
    if os.name!='nt':
        return Path(tempfile.mkdtemp(prefix=prefix,dir=parent))
    # Windows Python's mode=0700 creates a protected owner-only DACL. A rename
    # retains it, so another user allowed by the chosen parent cannot read the
    # finished export. Normal mkdir inherits that parent's access policy. Names
    # remain unpredictable, exclusive and on the same volume for publication.
    for _ in range(100):
        stage=parent/(prefix+secrets.token_hex(16))
        try:stage.mkdir(mode=0o777)
        except FileExistsError:continue
        return stage
    raise FileExistsError('Could not allocate an exclusive export staging directory')
