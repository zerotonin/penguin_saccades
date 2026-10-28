# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — __init__                                     ║
# ║  « saccadic gaze strategy of swimming penguins »                 ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Public API and version.                                         ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Saccadic gaze strategy analysis for swimming penguins."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("penguin_saccades")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
