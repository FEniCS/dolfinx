# -*- coding: utf-8 -*-
# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tools for interfacing with the GMSH Python API"""

from dolfinx.cpp.io import perm_gmsh
from dolfinx.mesh.gmsh.utils import extract_gmsh_geometry, extract_gmsh_topology_and_markers, ufl_mesh_from_gmsh

__all__ = ["extract_gmsh_geometry", "extract_gmsh_topology_and_markers", "ufl_mesh_from_gmsh", "perm_gmsh"]
