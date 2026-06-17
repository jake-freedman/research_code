"""
Path utilities for cross-machine compatibility.

config.py is .gitignored (each machine has its own DATA_DIR), so this file
provides helpers that depend on DATA_DIR at runtime without hard-coding it.
"""

from pathlib import Path
from config import DATA_DIR


def local_path(copied_path: str) -> str:
    """Convert a Windows path from any machine to the local equivalent.

    Replaces the C:\\Users\\<other_username> prefix with the local
    C:\\Users\\<local_username> (taken from DATA_DIR in config.py).
    This lets paths copied from another machine or user account resolve
    correctly here, regardless of drive letter or username differences.

    If no 'Users' component is found in the given path it is returned
    unchanged (it may already be a valid local path).

    Usage::

        from path_utils import local_path
        DATA_FILE = local_path(r'C:\\Users\\other_user\\OneDrive - UCB-O365\\...\\myfile.npz')
    """
    p = Path(copied_path.strip().strip('"'))
    p_parts = p.parts

    # Find 'Users' in the given path
    p_users_idx = next(
        (i for i, part in enumerate(p_parts) if part.lower() == 'users'),
        None,
    )
    if p_users_idx is None or p_users_idx + 1 >= len(p_parts):
        return str(p)

    # Find 'Users' in the local DATA_DIR to get the local username prefix
    local_parts = Path(DATA_DIR).parts
    local_users_idx = next(
        (i for i, part in enumerate(local_parts) if part.lower() == 'users'),
        None,
    )
    if local_users_idx is None or local_users_idx + 1 >= len(local_parts):
        return str(p)

    # Replace drive + Users + <username> with local equivalents; keep the rest
    local_prefix = local_parts[:local_users_idx + 2]   # e.g. ('C:\\', 'Users', 'acous')
    remainder    = p_parts[p_users_idx + 2:]            # everything after <other_username>

    result = Path(*local_prefix, *remainder) if remainder else Path(*local_prefix)
    return str(result)
