"""
Path utilities for cross-machine compatibility.

config.py is .gitignored (each machine has its own DATA_DIR / MEDIA_DIR), so
this file provides helpers that depend on those roots at runtime without
hard-coding them.
"""

from pathlib import Path
from config import DATA_DIR, MEDIA_DIR

_ROOTS = [DATA_DIR, MEDIA_DIR]


def local_path(copied_path: str) -> str:
    """Convert a Windows 'Copy as path' string from any machine to the local equivalent.

    Tries DATA_DIR and MEDIA_DIR in turn, picks the root that yields the
    longest suffix match against the given path (case-insensitive), then
    replaces the matched prefix with the local root and appends the remainder.

    Usage::

        from path_utils import local_path
        OUT_DIR = local_path(r'"C:\\Users\\other_user\\OneDrive - UCB-O365\\...\\data\\myrun"')
    """
    p = Path(copied_path.strip().strip('"'))
    p_parts = p.parts

    best_result = None
    best_suffix_len = 0

    for root in _ROOTS:
        root_path  = Path(root)
        root_parts = root_path.parts

        for suffix_len in range(len(root_parts), 0, -1):
            suffix      = root_parts[len(root_parts) - suffix_len:]
            window_size = len(suffix)
            for i in range(len(p_parts) - window_size + 1):
                if all(a.lower() == b.lower()
                       for a, b in zip(p_parts[i:i + window_size], suffix)):
                    rel_parts = p_parts[i + window_size:]
                    result    = root_path.joinpath(*rel_parts) if rel_parts else root_path
                    if suffix_len > best_suffix_len:
                        best_suffix_len = suffix_len
                        best_result     = str(result)
                    break
            if best_suffix_len == len(root_parts):
                break

    if best_result is not None:
        return best_result

    raise ValueError(
        f"Could not locate any part of DATA_DIR or MEDIA_DIR in the given path.\n"
        f"  Given     : {p}\n"
        f"  DATA_DIR  : {DATA_DIR}\n"
        f"  MEDIA_DIR : {MEDIA_DIR}"
    )
