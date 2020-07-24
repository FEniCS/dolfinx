# Copyright (C) 2018 Chris N Richardson
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfinx import FunctionSpace, UnitCubeMesh, UnitSquareMesh
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
