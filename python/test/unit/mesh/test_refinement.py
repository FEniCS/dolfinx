# Copyright (C) 2018-2020 Chris N Richardson, Jack S. Hale
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np

from dolfinx import FunctionSpace, MeshTags, UnitCubeMesh, UnitSquareMesh
from dolfinx.cpp.mesh import GhostMode
from dolfinx.mesh import refine
from mpi4py import MPI


def test_RefineUnitSquareMesh():
    """Refine mesh of unit square."""
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 5, 7, ghost_mode=GhostMode.none)
    mesh.topology.create_entities(1)
    mesh = refine(mesh, redistribute=False)
    assert mesh.topology.index_map(0).size_global == 165
    assert mesh.topology.index_map(2).size_global == 280


def test_NoOpRefineMarkedUnitSquareMesh():
    """Refine a mesh of unit square with markers set to zero.

    Should produce a mesh with the same number of cells as the unrefined mesh, i.e. a no-op.
    """
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 5, 7, ghost_mode=GhostMode.none)
    mesh.topology.create_entities(1)

    markers = np.zeros(mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int8)
    indices = np.arange(0, mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32)

    markers_mesh_tags = MeshTags(mesh, mesh.topology.dim, indices, markers)
    refined_mesh = refine(mesh, cell_markers=markers_mesh_tags)

    for i in range(0, 3):
        assert mesh.topology.index_map(i).size_global == refined_mesh.topology.index_map(i).size_global


def test_RefineUnitCubeMesh_repartition():
    """Refine mesh of unit cube."""
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 5, 7, 9, ghost_mode=GhostMode.none)
    mesh.topology.create_entities(1)
    mesh = refine(mesh, redistribute=True)
    assert mesh.topology.index_map(0).size_global == 3135
    assert mesh.topology.index_map(3).size_global == 15120

    mesh = UnitCubeMesh(MPI.COMM_WORLD, 5, 7, 9, ghost_mode=GhostMode.shared_facet)
    mesh.topology.create_entities(1)
    mesh = refine(mesh, redistribute=True)
    assert mesh.topology.index_map(0).size_global == 3135
    assert mesh.topology.index_map(3).size_global == 15120

    Q = FunctionSpace(mesh, ("CG", 1))
    assert(Q)


def test_RefineUnitCubeMesh_keep_partition():
    """Refine mesh of unit cube."""
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 5, 7, 9, ghost_mode=GhostMode.none)
    mesh.topology.create_entities(1)
    mesh = refine(mesh, redistribute=False)
    assert mesh.topology.index_map(0).size_global == 3135
    assert mesh.topology.index_map(3).size_global == 15120
    Q = FunctionSpace(mesh, ("CG", 1))
    assert(Q)


def xtest_refinement_gdim():
    """Test that 2D refinement is still 2D"""
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 3, 4, ghost_mode=GhostMode.none)
    mesh2 = refine(mesh, redistribute=True)
    assert mesh.geometry.dim == mesh2.geometry.dim
