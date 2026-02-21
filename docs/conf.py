"""Sphinx configuration for the hyperscanning pipeline docs."""

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

project = "Hyperscanning EEG"
author = "Hiroyoshi YAMASAKI"
copyright = "MIT"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "numpydoc",
]

try:
    import sphinx_autodoc_typehints  # noqa: F401
except Exception:
    _TYPEHINTS_AVAILABLE = False
else:
    _TYPEHINTS_AVAILABLE = True
    extensions.append("sphinx_autodoc_typehints")

try:
    import myst_parser  # noqa: F401
except Exception:
    _MYST_AVAILABLE = False
else:
    _MYST_AVAILABLE = True
    extensions.append("myst_parser")

try:
    import sphinx_copybutton  # noqa: F401
except Exception:
    _COPYBUTTON_AVAILABLE = False
else:
    _COPYBUTTON_AVAILABLE = True
    extensions.append("sphinx_copybutton")

try:
    import sphinx_design  # noqa: F401
except Exception:
    _DESIGN_AVAILABLE = False
else:
    _DESIGN_AVAILABLE = True
    extensions.append("sphinx_design")

try:
    import sphinxcontrib.mermaid  # noqa: F401
except Exception:
    _MERMAID_AVAILABLE = False
else:
    _MERMAID_AVAILABLE = True
    extensions.append("sphinxcontrib.mermaid")

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build"]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "groupwise"

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "mne": ("https://mne.tools/stable", None),
}

try:
    import pydata_sphinx_theme  # noqa: F401
except Exception:
    html_theme = "alabaster"
    html_theme_options = {}
else:
    html_theme = "pydata_sphinx_theme"
    html_theme_options = {
        "navigation_depth": 3,
        "show_toc_level": 2,
        "show_nav_level": 2,
        "navbar_align": "left",
    }

html_static_path: list[str] = []

# Render Mermaid in the browser so we don't require the CLI (mmdc).
mermaid_output_format = "raw"

if os.getenv("SPHINX_DEBUG"):
    nitpicky = True

autodoc_mock_imports = [
    "mne",
    "numpy",
    "pandas",
    "h5py",
    "matplotlib",
]
