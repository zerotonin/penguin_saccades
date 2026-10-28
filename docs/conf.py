# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — docs/conf.py                                 ║
# ║  « Sphinx configuration: autodoc + Napoleon + RTD theme »        ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Builds the API reference from Google-style docstrings.          ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Sphinx build configuration for the penguin_saccades documentation."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "penguin_saccades"
author = "Bart R. H. Geurten, Anne Oepen"
copyright = "2026, University of Otago, Department of Zoology"

try:
    release = _pkg_version("penguin_saccades")
except PackageNotFoundError:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False
autodoc_mock_imports = ["cv2", "rerandomstats", "tables"]
autodoc_default_options = {"members": True, "undoc-members": True, "show-inheritance": True}
autodoc_typehints = "description"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_title = f"penguin_saccades {version}"
