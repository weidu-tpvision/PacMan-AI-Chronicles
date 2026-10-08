"""Shared helper for locating checkpoint / weight files."""

import os

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS_DIR = os.path.join(_PKG_ROOT, "rl", "weights")


def resolve_weights_path(filename: str) -> str:
    """Return an absolute path for `filename`, searching rl/weights/, CWD and repo root.

    Absolute paths and existing relative paths are returned unchanged (absolutised).
    If nothing exists, the canonical rl/weights/<filename> location is returned.
    """
    if os.path.isabs(filename):
        return filename
    candidates = [
        os.path.join(WEIGHTS_DIR, filename),
        os.path.join(os.getcwd(), filename),
        os.path.join(_PKG_ROOT, filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.join(WEIGHTS_DIR, filename)
