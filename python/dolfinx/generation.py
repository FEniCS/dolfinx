# Copyright (C) 2017-2020 Chris N. Richardson, Michal Habera and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Simple built-in mesh generation module"""

import typing

import numpy

import ufl
from dolfinx import cpp as _cpp
from dolfinx.cpp.generation import DiagonalType  # noqa
from dolfinx.mesh import CellType, GhostMode, Mesh

from mpi4py import MPI as _MPI

__all__ = [
    "IntervalMesh", "UnitIntervalMesh", "RectangleMesh", "UnitSquareMesh",
    "BoxMesh", "UnitCubeMesh"
]


def IntervalMesh(comm: _MPI.Comm, nx: int, points: list, ghost_mode=GhostMode.shared_facet,
                 partitioner=_cpp.mesh.create_cell_partitioner()):
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
        The ghost mode used in the mesh partitioning. Options are
        `GhostMode.none' and `GhostMode.shared_facet`.
    partitioner
        Partitioning function to use for determining the parallel
        distribution of cells across MPI ranks

    """
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "interval", 1))
    mesh = _cpp.generation.create_interval_mesh(comm, nx, points, ghost_mode, partitioner)
    return Mesh.from_cpp(mesh, domain)


def UnitIntervalMesh(comm: _MPI.Comm, nx: int, ghost_mode=GhostMode.shared_facet,
                     partitioner=_cpp.mesh.create_cell_partitioner()):
    """Create a mesh on the unit interval

    Parameters
    ----------
    comm
        MPI communicator
    nx
        Number of cells
    ghost_mode
        The ghost mode used in the mesh partitioning. Options are
        `GhostMode.none' and `GhostMode.shared_facet`.
    partitioner
        Partitioning function to use for determining the parallel
        distribution of cells across MPI ranks

    """
    return IntervalMesh(comm, nx, [0.0, 1.0], ghost_mode, partitioner)


def RectangleMesh(comm: _MPI.Comm, points: typing.List[numpy.array], n: list, cell_type=CellType.triangle,
                  ghost_mode=GhostMode.shared_facet, partitioner=_cpp.mesh.create_cell_partitioner(),
                  diagonal: DiagonalType = DiagonalType.right):
    """Create rectangle mesh

    Parameters
    ----------
    comm
        MPI communicator
    points
        Coordinates of the lower-left and upper-right corners of the
        rectangle
    n
        Number of cells in each direction
    cell_type
        The cell type
    ghost_mode
        The ghost mode used in the mesh partitioning
    partitioner
        Function that computes the parallel distribution of cells across
        MPI ranks
    diagonal
        Direction of diagonal of triangular meshes. The options are
        'left', 'right', 'crossed', 'left/right', 'right/left'

    """

    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell_type.name, 1))
    mesh = _cpp.generation.create_rectangle_mesh(comm, points, n, cell_type, ghost_mode, partitioner, diagonal)
    return Mesh.from_cpp(mesh, domain)


def UnitSquareMesh(comm: _MPI.Comm, nx: int, ny: int, cell_type=CellType.triangle,
                   ghost_mode=GhostMode.shared_facet, partitioner=_cpp.mesh.create_cell_partitioner(),
                   diagonal: DiagonalType = DiagonalType.right):
    """Create a mesh of a unit square

    Parameters
    ----------
    comm
        MPI communicator
    nx
        Number of cells in "x" direction
    ny
        Number of cells in "y" direction
    cell_type
        The cell type
    ghost_mode
        The ghost mode used in the mesh partitioning
    partitioner
        Function that computes the parallel distribution of cells across
        MPI ranks
    diagonal
        Direction of diagonal

    """
    return RectangleMesh(comm, [numpy.array([0.0, 0.0, 0.0]),
                                numpy.array([1.0, 1.0, 0.0])], [nx, ny], cell_type, ghost_mode,
                         partitioner, diagonal)


def BoxMesh(comm: _MPI.Comm, points: typing.List[numpy.array], n: list,
            cell_type=CellType.tetrahedron,
            ghost_mode=GhostMode.shared_facet,
            partitioner=_cpp.mesh.create_cell_partitioner()):
    """Create box mesh

    Parameters
    ----------
    comm
        MPI communicator
    points
        Coordinates of the 'lower-left' and 'upper-right' corners of the
        box
    n
        List of cells in each direction
    cell_type
        The cell type
    ghost_mode
        The ghost mode used in the mesh partitioning
    partitioner
        Function that computes the parallel distribution of cells across
        MPI ranks

    """
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell_type.name, 1))
    mesh = _cpp.generation.create_box_mesh(comm, points, n, cell_type, ghost_mode, partitioner)
    return Mesh.from_cpp(mesh, domain)


def UnitCubeMesh(comm: _MPI.Comm, nx: int, ny: int, nz: int, cell_type=CellType.tetrahedron,
                 ghost_mode=GhostMode.shared_facet, partitioner=_cpp.mesh.create_cell_partitioner()):
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
    cell_type
        The cell type
    ghost_mode
        The ghost mode used in the mesh partitioning
    partitioner
        Function that computes the parallel distribution of cells across
        MPI ranks

    """
    return BoxMesh(comm, [numpy.array([0.0, 0.0, 0.0]), numpy.array(
        [1.0, 1.0, 1.0])], [nx, ny, nz], cell_type, ghost_mode, partitioner)
