"""Root conftest.py — project-level pytest configuration.

Adds the user home directory to sys.path so that the Vectro library
(at ~/vectro/) is importable as ``vectro.python.interface`` without a
separate pip install step.

Vectro is a namespace package (no top-level __init__.py), so putting
~/  on sys.path is the correct mechanism:
    ~/vectro/python/__init__.py   → vectro.python is a regular subpackage
    ~/vectro/python/interface.py  → vectro.python.interface is importable

This file is loaded automatically by pytest on every run.
"""
from __future__ import annotations

import os
import sys

_home = os.path.expanduser("~")
if _home not in sys.path:
    sys.path.insert(0, _home)
