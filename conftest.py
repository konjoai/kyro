"""Root conftest.py — project-level pytest configuration.

Adds the vectro parent directory to sys.path so that the Vectro library
(at <VECTRO_PATH>/vectro/ or ~/vectro/ by default) is importable as
``vectro.python.interface`` without a separate pip install step.

Set VECTRO_PATH in CI to the directory containing the vectro checkout.
Vectro is a namespace package (no top-level __init__.py), so putting
the parent dir on sys.path is the correct mechanism:
    <parent>/vectro/python/__init__.py   → vectro.python is a regular subpackage
    <parent>/vectro/python/interface.py  → vectro.python.interface is importable

This file is loaded automatically by pytest on every run.
"""

from __future__ import annotations

import os
import sys

_vectro_parent = os.environ.get("VECTRO_PATH", os.path.expanduser("~"))
if _vectro_parent not in sys.path:
    sys.path.insert(0, _vectro_parent)
