# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import dolfinx
import basix
import ufl
import datetime
import ffcx

sys.path.insert(0, os.path.abspath("."))

import jupytext_process  # isort:skip


myst_heading_anchors = 3

jupytext_process.process()

# -- Project information -----------------------------------------------------

project = "DOLFINx"
now = datetime.datetime.now()
date = now.date()
copyright = f"{date.year}, FEniCS Project"
author = "FEniCS Project"

# TODO: automate version tagging?
# The full version, including alpha/beta/rc tags
release = dolfinx.cpp.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.mathjax",
    "sphinx_codeautolink",
    "sphinx.ext.viewcode",
    "sphinx_codeautolink",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "breathe",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = "nature"
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
html_static_path = []

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

breathe_projects = {"DOLFINx": "../xml/"}
breathe_default_project = "DOLFINx"
breathe_implementation_filename_extensions = [".c", ".cc", ".cpp"]
breathe_domain_by_extension = {
    "h": "cpp",
}

# Tell sphinx what the primary language being documented is.
primary_domain = "cpp"

# Tell sphinx what the pygments highlight language should be.
highlight_language = "cpp"

intersphinx_resolve_self = "dolfinx"
codeautolink_concat_default = True

# Could be reimplemented using packaging.version
basix_version = "main" if "dev0" in basix.__version__ else "v" + basix.__version__
ffcx_version = "main" if "dev0" in ffcx.__version__ else "v" + ffcx.__version__
ufl_version = "main" if "dev0" in ufl.__version__ else ufl.__version__


# Note that as of late 2025 pyvista and petsc4py only have docs for the latest
# releases.
intersphinx_mapping = {
    "basix": (
        f"https://docs.fenicsproject.org/basix/{basix_version}/python",
        None,
    ),
    "ffcx": (
        f"https://docs.fenicsproject.org/ffcx/{ffcx_version}",
        None,
    ),
    "ufl": (
        f"https://docs.fenicsproject.org/ufl/{ufl_version}",
        None,
    ),
}
