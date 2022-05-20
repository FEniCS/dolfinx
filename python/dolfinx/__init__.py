# Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Main module for DOLFINx"""

# flake8: noqa
import sys

from dolfinx import common
from dolfinx import cpp as _cpp
from dolfinx import fem, geometry, graph, io, jit, la, log, mesh, nls, plot
# Initialise logging
from dolfinx.common import (TimingType, git_commit_hash, has_debug, has_kahip,
                            has_parmetis, list_timings, timing)
# Import cpp modules
from dolfinx.cpp import __version__

_cpp.common.init_logging(sys.argv)
del _cpp, sys

def get_include(user=False):
    import os
    d = os.path.dirname(__file__)
    if os.path.exists(os.path.join(d, "wrappers")):
        # Package is installed
        return os.path.join(d, "wrappers")
    else:
        # Package is from a source directory
        return os.path.join(os.path.dirname(d), "src")


__all__ = [
    "fem", "common", "geometry", "graph", "io", "jit", "la", "log", "mesh", "nls", "plot",
    "TimingType", "git_commit_hash", "has_debug", "has_kahip", "has_parmetis", "list_timings",
    "timing"
]
