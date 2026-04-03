"""
Path utilities for cross-machine compatibility.

config.py is .gitignored (each machine has its own DATA_DIR), so this file
provides helpers that depend on DATA_DIR at runtime without hard-coding it.
"""

from pathlib import Path
from config import DATA_DIR


def local_path(copied_path: str) -> str:
    """Convert a Windows 'Copy as path' string from any machine to the local equivalent.

    Windows Explorer's "Copy as path" produces a quoted absolute path specific to
    that machine (e.g. a different username in C:\\Users\\<name>\\...). This function
    strips the quotes, then finds the longest trailing portion of DATA_DIR that
    appears as a contiguous sequence inside the given path (case-insensitive).
    This lets it work even when the leading components (drive, username) differ
    across machines.

    Usage::

        from path_utils import local_path
        DATA_FILE = local_path(r'"C:\\Users\\other_user\\OneDrive - UCB-O365\\...\\data\\myfile.csv"')
    """
    p = Path(copied_path.strip().strip('"'))
    data_dir = Path(DATA_DIR)
    data_parts = data_dir.parts
    p_parts = p.parts

    # Try matching suffixes of data_parts (longest first) against a window in p_parts.
    for suffix_len in range(len(data_parts), 0, -1):
        suffix = data_parts[len(data_parts) - suffix_len:]
        window_size = len(suffix)
        for i in range(len(p_parts) - window_size + 1):
            if all(a.lower() == b.lower() for a, b in zip(p_parts[i:i + window_size], suffix)):
                rel_parts = p_parts[i + window_size:]
                result = data_dir.joinpath(*rel_parts) if rel_parts else data_dir
                return str(result)

    raise ValueError(
        f"Could not locate any part of DATA_DIR structure in the given path.\n"
        f"  Given   : {p}\n"
        f"  DATA_DIR: {data_dir}"
    )
