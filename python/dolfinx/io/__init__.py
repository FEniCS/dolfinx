# Copyright (C) 2022 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tools for file input/output (IO)."""

from dolfinx import cpp as _cpp
from dolfinx.io import gmshio
from dolfinx.io.utils import VTKFile, XDMFFile, distribute_entity_data

__all__ = ["VTKFile", "XDMFFile", "distribute_entity_data", "gmshio"]

if _cpp.common.has_adios2:
    # VTXWriter requires ADIOS2
    from dolfinx.io.utils import VTXMeshPolicy, VTXWriter

    __all__ = [*__all__, "VTXWriter", "VTXMeshPolicy"]
