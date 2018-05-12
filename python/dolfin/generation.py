# -*- coding: utf-8 -*-
# Copyright (C) 2017 Chris N. Richardson
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Simple mesh generation module"""

import dolfin.fem
from dolfin.cpp.generation import IntervalMesh, RectangleMesh, BoxMesh
from dolfin.cpp.mesh import CellType, GhostMode


# FIXME: Remove, and use 'create' method?

def UnitIntervalMesh(comm, nx,
                     ghost_mode=GhostMode.none):
    """Create a mesh on the unit interval"""
    mesh = IntervalMesh.create(comm, nx, [0.0, 1.0], ghost_mode)
    mesh.geometry.coord_mapping = dolfin.fem.create_coordinate_map(mesh)
    return mesh


def UnitSquareMesh(comm, nx, ny, cell_type=CellType.Type.triangle,
                   ghost_mode=GhostMode.none, diagonal="right"):
    """Create a mesh of a unit square"""
    from dolfin.cpp.geometry import Point
    mesh = RectangleMesh.create(comm, [Point(0.0, 0.0), Point(1.0, 1.0)],
                                [nx, ny], cell_type, ghost_mode, diagonal)
    mesh.geometry.coord_mapping = dolfin.fem.create_coordinate_map(mesh)
    return mesh


def UnitCubeMesh(comm, nx, ny, nz, cell_type=CellType.Type.tetrahedron,
                 ghost_mode=GhostMode.none):
    """Create a mesh of a unit cube"""
    from dolfin.cpp.geometry import Point
    mesh = BoxMesh.create(comm, [Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0)],
                          [nx, ny, nz], cell_type, ghost_mode)
    mesh.geometry.coord_mapping = dolfin.fem.create_coordinate_map(mesh)
    return mesh
