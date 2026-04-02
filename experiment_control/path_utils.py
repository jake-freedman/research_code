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
    strips the quotes, locates the DATA_DIR folder structure within the given path
    by sliding a case-insensitive window match over its components, and rebuilds the
    path under the local DATA_DIR.

    Usage::

        from path_utils import local_path
        DATA_FILE = local_path(r'"C:\\Users\\acous\\OneDrive - UCB-O365\\...\\data\\myfile.csv"')
    """
    p = Path(copied_path.strip().strip('"'))
    data_dir = Path(DATA_DIR)
    data_parts = data_dir.parts
    p_parts = p.parts
    n = len(data_parts)

    for i in range(len(p_parts) - n + 1):
        if all(a.lower() == b.lower() for a, b in zip(p_parts[i:i + n], data_parts)):
            rel_parts = p_parts[i + n:]
            result = data_dir.joinpath(*rel_parts) if rel_parts else data_dir
            return str(result)

    raise ValueError(
        f"Could not locate DATA_DIR structure in the given path.\n"
        f"  Given   : {p}\n"
        f"  DATA_DIR: {data_dir}"
    )
