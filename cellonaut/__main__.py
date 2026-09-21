"""Keep direct-file and module launches on the same application entry path."""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

# Running this file directly does not give Python the package root that
# `python -m cellonaut` and installed launchers provide.
if __package__ in {None, ""}:
    sys.path[0] = str(Path(__file__).resolve().parent.parent)

from cellonaut.app import main


# Frozen multiprocessing workers re-import the entry module, so hand control to
# Python's frozen-process bootstrap before opening the GUI.
if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
