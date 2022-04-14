# Copyright (C) 2012 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tools for input/output (IO)."""

from dolfinx.io import gmsh
from dolfinx.cpp.io import distribute_entity_data  # noqa: F401
from dolfinx import cpp as _cpp
from dolfinx.io.io import VTKFile, XDMFFile

__all__ = ["gmsh", "distribute_entity_data", "FidesWriter", "VTKFile", "VTXWriter", "XDMFFile"]

if _cpp.common.has_adios2:
    # FidesWriter and VTXWriter require ADIOS2
    from dolfinx.io.io import FidesWriter, VTXWriter  # noqa: F401
    __all__ = __all__ + ["FidesWriter", "VTXWriter"]
