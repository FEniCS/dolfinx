# Copyright (C) 2022 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tools for file input/output (IO)."""

from dolfinx.common import has_adios2
from dolfinx.io import gmsh as gmsh
from dolfinx.io import gmsh as gmshio  # legacy compatibility.
from dolfinx.io import vtkhdf
from dolfinx.io.utils import VTKFile, XDMFFile, distribute_entity_data

__all__ = ["VTKFile", "XDMFFile", "distribute_entity_data", "gmsh", "gmshio", "vtkhdf"]

if has_adios2:
    # VTXWriter requires ADIOS2
    from dolfinx.io.utils import VTXMeshPolicy, VTXWriter

    __all__ += ["VTXMeshPolicy", "VTXWriter"]
