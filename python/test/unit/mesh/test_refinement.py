# Copyright (C) 2018-2021 Chris N Richardson and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
import pytest
from numpy import isclose, logical_and

import ufl
from dolfinx import cpp as _cpp
from dolfinx.fem import FunctionSpace, form
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.mesh import (CellType, DiagonalType, GhostMode,
                          compute_incident_entities, create_unit_cube,
                          create_unit_square, locate_entities,
                          locate_entities_boundary, meshtags, refine)

from mpi4py import MPI


def test_Refinecreate_unit_square():
    """Refine mesh of unit square."""
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 7, ghost_mode=GhostMode.none)
    mesh.topology.create_entities(1)
    mesh = refine(mesh, redistribute=False)
    assert mesh.topology.index_map(0).size_global == 165
    assert mesh.topology.index_map(2).size_global == 280


def test_Refinecreate_unit_cube_repartition():
    """Refine mesh of unit cube."""
    mesh = create_unit_cube(MPI.COMM_WORLD, 5, 7, 9, ghost_mode=GhostMode.none)
    mesh.topology.create_entities(1)
    mesh = refine(mesh, redistribute=True)
    assert mesh.topology.index_map(0).size_global == 3135
    assert mesh.topology.index_map(3).size_global == 15120

    mesh = create_unit_cube(MPI.COMM_WORLD, 5, 7, 9, ghost_mode=GhostMode.shared_facet)
    mesh.topology.create_entities(1)
    mesh = refine(mesh, redistribute=True)
    assert mesh.topology.index_map(0).size_global == 3135
    assert mesh.topology.index_map(3).size_global == 15120

    Q = FunctionSpace(mesh, ("Lagrange", 1))
    assert Q


def test_Refinecreate_unit_cube_keep_partition():
    """Refine mesh of unit cube."""
    mesh = create_unit_cube(MPI.COMM_WORLD, 5, 7, 9, ghost_mode=GhostMode.none)
    mesh.topology.create_entities(1)
    mesh = refine(mesh, redistribute=False)
    assert mesh.topology.index_map(0).size_global == 3135
    assert mesh.topology.index_map(3).size_global == 15120
    Q = FunctionSpace(mesh, ("Lagrange", 1))
    assert Q


def test_refine_create_form():
    """Check that forms can be assembled on refined mesh"""
    mesh = create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    mesh.topology.create_entities(1)
    mesh = refine(mesh, redistribute=True)

    V = FunctionSpace(mesh, ("Lagrange", 1))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    assemble_matrix(a)


def test_refinement_gdim():
    """Test that 2D refinement is still 2D"""
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 4, ghost_mode=GhostMode.none)
    mesh.topology.create_entities(1)
    mesh2 = refine(mesh, redistribute=True)
    assert mesh.geometry.dim == mesh2.geometry.dim


def test_sub_refine():
    """Test that refinement of a subset of edges works"""
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 4, diagonal=DiagonalType.left,
                              ghost_mode=GhostMode.none)
    mesh.topology.create_entities(1)

    def left_corner_edge(x, tol=1e-16):
        return logical_and(isclose(x[0], 0), x[1] < 1 / 4 + tol)

    edges = locate_entities_boundary(mesh, 1, left_corner_edge)
    if MPI.COMM_WORLD.size == 0:
        assert edges == 1

    mesh2 = refine(mesh, edges, redistribute=False)
    assert mesh.topology.index_map(2).size_global + 3 == mesh2.topology.index_map(2).size_global


def test_refine_from_cells():
    """Check user interface for using local cells to define edges"""
    Nx = 8
    Ny = 3
    assert Nx % 2 == 0
    mesh = create_unit_square(MPI.COMM_WORLD, Nx, Ny, diagonal=DiagonalType.left, ghost_mode=GhostMode.none)
    mesh.topology.create_entities(1)

    def left_side(x, tol=1e-16):
        return x[0] <= 0.5 + tol
    cells = locate_entities(mesh, mesh.topology.dim, left_side)
    if MPI.COMM_WORLD.size == 0:
        assert cells.__len__() == Nx * Ny
    edges = compute_incident_entities(mesh, cells, 2, 1)
    if MPI.COMM_WORLD.size == 0:
        assert edges.__len__() == Nx // 2 * (2 * Ny + 1) + (Nx // 2 + 1) * Ny
    mesh2 = refine(mesh, edges, redistribute=True)

    num_cells_global = mesh2.topology.index_map(2).size_global
    actual_cells = 3 * (Nx * Ny) + 3 * Ny + 2 * Nx * Ny
    assert num_cells_global == actual_cells


@pytest.mark.parametrize("tdim", [2, 3])
def test_refine_facet_meshtag(tdim):
    if tdim == 3:
        mesh = create_unit_cube(MPI.COMM_WORLD, 2, 3, 5, CellType.tetrahedron, GhostMode.none)
    else:
        mesh = create_unit_square(MPI.COMM_WORLD, 2, 5, CellType.triangle, GhostMode.none)
    mesh.topology.create_entities(tdim - 1)
    mesh.topology.create_connectivity(tdim - 1, tdim)
    mesh.topology.create_entities(1)
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)
    facet_indices = []
    for f in range(mesh.topology.index_map(tdim - 1).size_local):
        if len(f_to_c.links(f)) == 1:
            facet_indices += [f]
    meshtag = meshtags(mesh, tdim - 1, numpy.array(facet_indices, dtype=numpy.int32),
                       numpy.arange(len(facet_indices), dtype=numpy.int32))

    fine_mesh, parent_cell, parent_facet = _cpp.refinement.plaza_refine_data(
        mesh._cpp_object, False, _cpp.refinement.RefinementOptions.parent_cell_and_facet)
    fine_mesh.topology.create_entities(tdim - 1)

    new_meshtag = _cpp.refinement.transfer_facet_meshtag(meshtag, fine_mesh, parent_cell, parent_facet)

    assert len(new_meshtag.indices) == (tdim * 2 - 2) * len(meshtag.indices)

    # New tags should be on facets with one cell (i.e. exterior)
    fine_mesh.topology.create_connectivity(tdim - 1, tdim)
    new_f_to_c = fine_mesh.topology.connectivity(tdim - 1, tdim)
    for f in new_meshtag.indices:
        assert len(new_f_to_c.links(f)) == 1

    # Now mark all facets (including internal)
    facet_indices = numpy.arange(mesh.topology.index_map(tdim - 1).size_local)
    meshtag = meshtags(mesh, tdim - 1, numpy.array(facet_indices, dtype=numpy.int32),
                       numpy.arange(len(facet_indices), dtype=numpy.int32))

    new_meshtag = _cpp.refinement.transfer_facet_meshtag(meshtag, fine_mesh, parent_cell, parent_facet)

    assert len(new_meshtag.indices) == (tdim * 2 - 2) * len(meshtag.indices)


@pytest.mark.parametrize("tdim", [2, 3])
def test_refine_cell_meshtag(tdim):

    if tdim == 3:
        mesh = create_unit_cube(MPI.COMM_WORLD, 2, 3, 5, CellType.tetrahedron, GhostMode.none)
    else:
        mesh = create_unit_square(MPI.COMM_WORLD, 2, 5, CellType.triangle, GhostMode.none)

    mesh.topology.create_entities(1)

    cell_indices = numpy.arange(mesh.topology.index_map(tdim).size_local)
    meshtag = meshtags(mesh, tdim, numpy.array(cell_indices, dtype=numpy.int32),
                       numpy.arange(len(cell_indices), dtype=numpy.int32))

    fine_mesh, parent_cell, parent_facet = _cpp.refinement.plaza_refine_data(
        mesh._cpp_object, False, _cpp.refinement.RefinementOptions.parent_cell_and_facet)

    new_meshtag = _cpp.refinement.transfer_cell_meshtag(meshtag, fine_mesh, parent_cell)

    assert sum(new_meshtag.values) == (tdim * 4 - 4) * sum(meshtag.values)
    assert len(new_meshtag.indices) == (tdim * 4 - 4) * len(meshtag.indices)
