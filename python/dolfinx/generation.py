# Copyright (C) 2017-2020 Chris N. Richardson, Michal Habera and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Simple built-in mesh generation module"""

import typing

import numpy
import ufl

from dolfinx import cpp

__all__ = [
    "IntervalMesh", "UnitIntervalMesh", "RectangleMesh", "UnitSquareMesh",
    "BoxMesh", "UnitCubeMesh"
]


def IntervalMesh(comm, nx: int, points: list, ghost_mode=cpp.mesh.GhostMode.shared_facet,
                 partitioner=cpp.mesh.partition_cells_graph):
    """Create an interval mesh

    Parameters
    ----------
    comm
        MPI communicator
    nx
        Number of cells
    point
        Coordinates of the end points
    ghost_mode

    """
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "interval", 1))
    mesh = cpp.generation.create_interval_mesh(comm, nx, points, ghost_mode, partitioner)
    domain._ufl_cargo = mesh
    mesh._ufl_domain = domain
    return mesh


def UnitIntervalMesh(comm, nx, ghost_mode=cpp.mesh.GhostMode.shared_facet,
                     partitioner=cpp.mesh.partition_cells_graph):
    """Create a mesh on the unit interval

    Parameters
    ----------
    comm
        MPI communicator
    nx
        Number of cells

    """
    return IntervalMesh(comm, nx, [0.0, 1.0], ghost_mode)


def RectangleMesh(comm, points: typing.List[numpy.array], n: list, cell_type=cpp.mesh.CellType.triangle,
                  ghost_mode=cpp.mesh.GhostMode.shared_facet, partitioner=cpp.mesh.partition_cells_graph,
                  diagonal: str = "right"):
    """Create rectangle mesh

    Parameters
    ----------
    comm
        MPI communicator
    points
        List of `Points` representing vertices
    n
        List of number of cells in each direction
    diagonal
        Direction of diagonal

    """
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell_type.name, 1))
    mesh = cpp.generation.create_rectangle_mesh(comm, points, n, cell_type, ghost_mode, partitioner, diagonal)
    domain._ufl_cargo = mesh
    mesh._ufl_domain = domain
    return mesh


def UnitSquareMesh(comm, nx, ny, cell_type=cpp.mesh.CellType.triangle,
                   ghost_mode=cpp.mesh.GhostMode.shared_facet,
                   partitioner=cpp.mesh.partition_cells_graph, diagonal="right"):
    """Create a mesh of a unit square

    Parameters
    ----------
    comm
        MPI communicator
    nx
        Number of cells in "x" direction
    ny
        Number of cells in "y" direction
    diagonal
        Direction of diagonal

    """
    return RectangleMesh(comm, [numpy.array([0.0, 0.0, 0.0]),
                                numpy.array([1.0, 1.0, 0.0])], [nx, ny], cell_type, ghost_mode,
                         partitioner, diagonal)


def BoxMesh(comm, points: typing.List[numpy.array], n: list,
            cell_type=cpp.mesh.CellType.tetrahedron,
            ghost_mode=cpp.mesh.GhostMode.shared_facet,
            partitioner=cpp.mesh.partition_cells_graph):
    """Create box mesh

    Parameters
    ----------
    comm
        MPI communicator
    points
        List of points representing vertices
    n
        List of cells in each direction

    """
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell_type.name, 1))
    mesh = cpp.generation.create_box_mesh(comm, points, n, cell_type, ghost_mode, partitioner)
    domain._ufl_cargo = mesh
    mesh._ufl_domain = domain
    return mesh


def UnitCubeMesh(comm, nx, ny, nz, cell_type=cpp.mesh.CellType.tetrahedron,
                 ghost_mode=cpp.mesh.GhostMode.shared_facet, partitioner=cpp.mesh.partition_cells_graph):
    """Create a mesh of a unit cube

    Parameters
    ----------
    comm
        MPI communicator
    nx
        Number of cells in "x" direction
    ny
        Number of cells in "y" direction
    nz
        Number of cells in "z" direction

    """
    return BoxMesh(comm, [numpy.array([0.0, 0.0, 0.0]), numpy.array(
        [1.0, 1.0, 1.0])], [nx, ny, nz], cell_type, ghost_mode, partitioner)
