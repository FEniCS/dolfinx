# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import datetime as dt
import logging
import os
import re
import sys
import warnings

from breathe.parser import ParseWarning
from sphinx.util import logging as sphinx_logging

import basix
import dolfinx
import ffcx
import ufl

sys.path.insert(0, os.path.abspath("."))

import jupytext_process  # isort:skip


myst_heading_anchors = 3

jupytext_process.process()

# -- Project information -----------------------------------------------------

project = "DOLFINx C++"
now = dt.datetime.now()
date = now.date()
copyright = f"{date.year}, FEniCS Project"
author = "FEniCS Project"

version = dolfinx.cpp.__version__
release = version

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
html_theme = "pydata_sphinx_theme"
html_title = project

# Version switcher. FEniCS/docs generates the version list from the
# published documentation. CI sets DOCS_VERSION to the branch or tag
# name the documentation is published under. The list has one entry per
# release, shared by its .postN builds.
html_theme_options = {
    "switcher": {
        "json_url": "https://docs.fenicsproject.org/dolfinx/switcher-cpp.json",
        "version_match": re.sub(r"\.post\d+$", "", os.environ.get("DOCS_VERSION", "main")),
    },
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    # The version list is fetched by the browser; don't fetch it at build time
    "check_switcher": False,
    "show_version_warning_banner": True,
}

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

# Enable nitpicky mode so an unresolvable :ref:/:doc:/cpp:* target
# surfaces as a warning, which fails the CI build (-W). Breathe links
# every identifier-shaped token in a C++ signature, so a type used in
# the documented API must itself be documented on one of these pages.
# The categories below cannot be resolved and are ignored.
nitpicky = True

nitpick_ignore_regex = [
    # Doxygen's own internal directory-sharded anchor/file IDs
    # (e.g. 'da/dfe/namespacedolfinx_1_1MPI') leak through as :ref:
    # targets when Breathe cannot map a Doxygen \ref/@ref tag onto a
    # real Sphinx label (usually a \ref to an overload not rendered on
    # this particular page). No hand-written label in this project
    # uses this naming scheme, so the pattern cannot mask a genuine
    # broken :ref:/:doc: link elsewhere.
    ("ref", r"^[0-9a-f]{2}/[0-9a-f]{3}/[\w]+$"),
    # External C/C++ libraries (MPI, PETSc, SLEPc, HDF5, pugixml,
    # ADIOS2, Boost) and ffcx-generated struct types: not part of
    # DOLFINx's own Doxygen project, and none publish a C++ API
    # inventory to intersphinx against.
    (
        "cpp:identifier",
        r"^(MPI_(Comm|Request|Datatype)"
        r"|Petsc(Scalar|Int|Real|ErrorCode)|Vec|Mat(NullSpace)?"
        r"|KSP(ConvergedReason)?|EPS(ConvergedReason)?|IS"
        r"|InsertMode|ISLocalToGlobalMapping"
        r"|hid_t|herr_t|u?int(8|16|32|64)_t|int_t)$",
    ),
    ("cpp:identifier", r"^pugi(::xml_node)?$"),
    ("cpp:identifier", r"^adios2(::(Dims|IO|Engine|Variable<\w+>|Attribute<\w+>))?$"),
    (
        "cpp:identifier",
        r"^boost(::multiprecision(::cpp_bin_float_double_extended)?"
        r"|::unordered_flat_map<.*>)?$",
    ),
    ("cpp:identifier", r"^ufcx_(form|expression)$"),
    # Members of `md`, the namespace alias for the external mdspan
    # implementation.
    ("cpp:identifier", r"^md::\w+$"),
    # A member type of a template parameter, which has no target.
    ("cpp:identifier", r"^container_type::value_type$"),
    # Correct references that the C++ domain cannot resolve because a
    # function parameter has the same name as the leading namespace or
    # type, e.g. `mesh::GhostMode` in `refine(const mesh::Mesh<T>& mesh,
    # ..., mesh::GhostMode ghost_mode = mesh::GhostMode::none)`. The
    # domain looks up the qualified name inside the parameter, whereas
    # C++ ignores non-types when looking up a name followed by `::`.
    (
        "cpp:identifier",
        r"^(mesh::(EntityMap|GhostMode(::none)?)|strategy::speed)$",
    ),
    # A bare mention of one of DOLFINx's/basix's own namespaces (or a
    # path of nothing but namespaces), with no member name after it, or
    # `dolfinx::scalar` (a real C++20 concept, common/types.h, used as a
    # template constraint): Breathe's signature linker tries to
    # cross-reference the leading namespace-qualifier segment of a
    # qualified name, or a concept used as a constraint, as if it were
    # itself a standalone target it should be able to look up locally.
    (
        "cpp:identifier",
        r"^(dolfinx|fem|mesh|graph|la|common|io|refinement|geometry|nls"
        r"|basix|scotch|md"
        r"|dolfinx::(fem(::petsc)?|graph(::(kahip|parmetis|scotch))?"
        r"|MPI(::tag(::consensus_nbx)?)?|scalar)"
        r"|basix::(cell(::type)?|maps(::type)?))$",
    ),
    # A concrete instantiation of one of DOLFINx's own class/function
    # templates (e.g. `mesh::Mesh<T>`, `Form<T, U>`), which Breathe
    # cannot cross-reference generically. A typo essentially never
    # happens to also be syntactically valid template-instantiation
    # text, so this cannot mask a genuine broken reference.
    ("cpp:identifier", r"^[\w:]+<.*>[\w:]*$"),
]

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

# "Unparseable C++ cross-reference" has no (type, target) pair to match
# via nitpick_ignore_regex: it is raised directly by the C++ domain's
# signature parser (sphinx.domains.cpp), not by reference resolution,
# for a default template argument it cannot parse as a type expression
# (e.g. the SFINAE-style `scalar_value_t<T>`) or a bare
# namespace-qualified prefix fragment. Silence this catalogued,
# closed set; any other unparseable cross-reference still warns.
_UNPARSEABLE_CPP_XREF_RE = re.compile(
    r"^:(fem(::petsc)?|graph(::(kahip|parmetis|scotch))?"
    r"|MPI(::tag(::consensus_nbx)?)?|scalar(_value_t<.*>)?)$"
)


class _SuppressKnownUnparseableCppXrefs(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        prefix = "Unparseable C++ cross-reference: "
        if msg.startswith(prefix):
            target = msg[len(prefix) :].split("\n", 1)[0].strip("'\"")
            if _UNPARSEABLE_CPP_XREF_RE.match(target):
                return False
        return True


# Breathe reports problems in the Doxygen XML, e.g. elements it does not
# recognise, as Python warnings, which bypass Sphinx and so -W. Report
# them through Sphinx instead. Newer Doxygen versions add a
# `conceptparts` element holding a concept's definition, which Breathe
# does not parse; it renders a concept's name and description only.
_IGNORED_BREATHE_PARSE_WARNING_RE = re.compile(r'unexpected element "conceptparts"$')
_logger = sphinx_logging.getLogger("dolfinx.conf")
_showwarning = warnings.showwarning


def _report_breathe_parse_warning(message, category, filename, lineno, file=None, line=None):
    if not issubclass(category, ParseWarning):
        _showwarning(message, category, filename, lineno, file, line)
    elif not _IGNORED_BREATHE_PARSE_WARNING_RE.search(str(message)):
        _logger.warning(f"Breathe: {message}")


def setup(app):
    warnings.showwarning = _report_breathe_parse_warning
    # sphinx.util.logging.getLogger prepends its own "sphinx." namespace
    # onto a module's __name__, so the real stdlib logger name here is
    # "sphinx.sphinx.domains.cpp", not "sphinx.domains.cpp".
    logging.getLogger("sphinx.sphinx.domains.cpp").addFilter(_SuppressKnownUnparseableCppXrefs())
