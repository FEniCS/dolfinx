# Copyright (C) 2018-2021 Chris N Richardson and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from numpy import isclose, logical_and

import ufl
from dolfinx.fem import FunctionSpace, assemble_matrix
from dolfinx.generation import DiagonalType, UnitCubeMesh, UnitSquareMesh
from dolfinx.mesh import (GhostMode, compute_incident_entities,
                          locate_entities, locate_entities_boundary, refine)

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

    Q = FunctionSpace(mesh, ("Lagrange", 1))
    assert Q


def test_RefineUnitCubeMesh_keep_partition():
    """Refine mesh of unit cube."""
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 5, 7, 9, ghost_mode=GhostMode.none)
    mesh.topology.create_entities(1)
    mesh = refine(mesh, redistribute=False)
    assert mesh.topology.index_map(0).size_global == 3135
    assert mesh.topology.index_map(3).size_global == 15120
    Q = FunctionSpace(mesh, ("Lagrange", 1))
    assert Q


def test_refine_create_form():
    """Check that forms can be assembled on refined mesh"""
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 3, 3, 3)
    mesh.topology.create_entities(1)
    mesh = refine(mesh, redistribute=True)

    V = FunctionSpace(mesh, ("Lagrange", 1))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    assemble_matrix(a)


def test_refinement_gdim():
    """Test that 2D refinement is still 2D"""
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 3, 4, ghost_mode=GhostMode.none)
    mesh.topology.create_entities(1)
    mesh2 = refine(mesh, redistribute=True)
    assert mesh.geometry.dim == mesh2.geometry.dim


def test_sub_refine():
    """Test that refinement of a subset of edges works"""
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 3, 4, diagonal=DiagonalType.left,
                          ghost_mode=GhostMode.none)
    mesh.topology.create_entities(1)

    def left_corner_edge(x, tol=1e-16):
        return logical_and(isclose(x[0], 0), x[1] < 1 / 4 + tol)

    edges = locate_entities_boundary(mesh, 1, left_corner_edge)
    if MPI.COMM_WORLD.size == 0:
        assert(edges == 1)

    mesh2 = refine(mesh, edges, redistribute=False)
    assert(mesh.topology.index_map(2).size_global + 3 == mesh2.topology.index_map(2).size_global)


def test_refine_from_cells():
    """Check user interface for using local cells to define edges"""
    Nx = 8
    Ny = 3
    assert(Nx % 2 == 0)
    mesh = UnitSquareMesh(MPI.COMM_WORLD, Nx, Ny, diagonal=DiagonalType.left, ghost_mode=GhostMode.none)
    mesh.topology.create_entities(1)

    def left_side(x, tol=1e-16):
        return x[0] <= 0.5 + tol
    cells = locate_entities(mesh, mesh.topology.dim, left_side)
    if MPI.COMM_WORLD.size == 0:
        assert(cells.__len__() == Nx * Ny)
    edges = compute_incident_entities(mesh, cells, 2, 1)
    if MPI.COMM_WORLD.size == 0:
        assert(edges.__len__() == Nx // 2 * (2 * Ny + 1) + (Nx // 2 + 1) * Ny)
    mesh2 = refine(mesh, edges, redistribute=True)

    num_cells_global = mesh2.topology.index_map(2).size_global
    actual_cells = 3 * (Nx * Ny) + 3 * Ny + 2 * Nx * Ny
    assert(num_cells_global == actual_cells)
