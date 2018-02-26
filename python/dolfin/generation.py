# Copyright (C) 2017 Chris N. Richardson
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Simple mesh generation module"""

from dolfin.cpp.generation import IntervalMesh, RectangleMesh, BoxMesh
from dolfin.cpp.mesh import CellType


# FIXME: Remove, and use 'create' method?

def UnitIntervalMesh(comm, nx):
    """Create a mesh on the unit interval"""
    return IntervalMesh.create(comm, nx, [0.0, 1.0])


def UnitSquareMesh(comm, nx, ny, cell_type=CellType.Type.triangle):
    """Create a mesh of a unit square"""
    from dolfin.cpp.geometry import Point
    return RectangleMesh.create(comm, [Point(0.0, 0.0), Point(1.0, 1.0)],
                                [nx, ny], cell_type)


def UnitCubeMesh(comm, nx, ny, nz, cell_type=CellType.Type.tetrahedron):
    """Create a mesh of a unit cube"""
    from dolfin.cpp.geometry import Point
    return BoxMesh.create(comm, [Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0)],
                          [nx, ny, nz], cell_type)
