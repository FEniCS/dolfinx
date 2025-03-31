# Copyright (C) 2018-2024 Chris N Richardson and Jørgen S. Dokken

# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest
from numpy import isclose

import ufl
from dolfinx.cpp.mesh import create_cell_partitioner
from dolfinx.fem import assemble_matrix, form, functionspace
from dolfinx.mesh import (
    CellType,
    DiagonalType,
    GhostMode,
    RefinementOption,
    compute_incident_entities,
    create_unit_cube,
    create_unit_square,
    locate_entities,
    locate_entities_boundary,
    meshtags,
    refine,
    transfer_meshtag,
)


def test_refine_create_unit_square():
    """Refine mesh of unit square."""
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 7, ghost_mode=GhostMode.none)
    mesh.topology.create_entities(1)
    mesh_refined, _, _ = refine(mesh)
    assert mesh_refined.topology.index_map(0).size_global == 165
    assert mesh_refined.topology.index_map(2).size_global == 280

    # Test that 2D refinement is still 2D
    assert mesh.geometry.dim == mesh_refined.geometry.dim


@pytest.mark.parametrize("ghost_mode", [GhostMode.none, GhostMode.shared_facet])
@pytest.mark.parametrize("redistribute", [True, False])
def test_refine_create_unit_cube(ghost_mode, redistribute):
    """Refine mesh of unit cube."""
    mesh = create_unit_cube(MPI.COMM_WORLD, 5, 7, 9, ghost_mode=ghost_mode)
    mesh.topology.create_entities(1)
    mesh, _, _ = refine(mesh, partitioner=create_cell_partitioner(ghost_mode), redistribute=True)
    assert mesh.topology.index_map(0).size_global == 3135
    assert mesh.topology.index_map(3).size_global == 15120

    Q = functionspace(mesh, ("Lagrange", 1))
    assert Q


def test_refine_create_form():
    """Check that forms can be assembled on refined mesh"""
    mesh = create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    mesh.topology.create_entities(1)
    mesh, _, _ = refine(mesh)
    V = functionspace(mesh, ("Lagrange", 1))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    assemble_matrix(a)


def test_sub_refine():
    """Test that refinement of a subset of edges works"""
    msh = create_unit_square(
        MPI.COMM_WORLD, 3, 4, diagonal=DiagonalType.left, ghost_mode=GhostMode.none
    )
    msh.topology.create_entities(1)

    def left_corner_edge(x, tol=1e-7):
        return isclose(x[0], 0) & (x[1] < 1 / 4 + tol)

    edges = locate_entities_boundary(msh, 1, left_corner_edge)
    if MPI.COMM_WORLD.size == 1:
        assert edges == 1

    msh1, _, _ = refine(msh, edges)
    assert msh.topology.index_map(2).size_global + 3 == msh1.topology.index_map(2).size_global


def test_refine_from_cells():
    """Check user interface for using local cells to define edges"""
    Nx, Ny = 8, 3
    assert Nx % 2 == 0
    msh = create_unit_square(
        MPI.COMM_WORLD, Nx, Ny, diagonal=DiagonalType.left, ghost_mode=GhostMode.none
    )
    msh.topology.create_entities(1)

    def left_side(x, tol=1e-16):
        return x[0] <= 0.5 + tol

    cells = locate_entities(msh, msh.topology.dim, left_side)
    if MPI.COMM_WORLD.size == 1:
        assert len(cells) == Nx * Ny
    edges = compute_incident_entities(msh.topology, cells, 2, 1)
    if MPI.COMM_WORLD.size == 1:
        assert len(edges) == Nx // 2 * (2 * Ny + 1) + (Nx // 2 + 1) * Ny
    mesh2, _, _ = refine(msh, edges)

    num_cells_global = mesh2.topology.index_map(2).size_global
    actual_cells = 3 * (Nx * Ny) + 3 * Ny + 2 * Nx * Ny
    assert num_cells_global == actual_cells


@pytest.mark.parametrize("tdim", [2, 3])
@pytest.mark.parametrize(
    "refine_plaza_wrapper",
    [
        lambda msh: refine(msh, option=RefinementOption.parent_cell_and_facet),
        lambda msh: refine(
            msh,
            edges=np.arange(msh.topology.index_map(1).size_local),
            option=RefinementOption.parent_cell_and_facet,
        ),
    ],
)
def test_refine_facet_meshtag(tdim, refine_plaza_wrapper):
    if tdim == 3:
        msh = create_unit_cube(
            MPI.COMM_WORLD, 2, 3, 5, CellType.tetrahedron, ghost_mode=GhostMode.none
        )
    else:
        msh = create_unit_square(MPI.COMM_WORLD, 2, 5, CellType.triangle, ghost_mode=GhostMode.none)
    msh.topology.create_entities(tdim - 1)
    msh.topology.create_connectivity(tdim - 1, tdim)
    msh.topology.create_entities(1)
    f_to_c = msh.topology.connectivity(tdim - 1, tdim)
    facet_indices = []
    for f in range(msh.topology.index_map(tdim - 1).size_local):
        if len(f_to_c.links(f)) == 1:
            facet_indices += [f]
    meshtag = meshtags(
        msh,
        tdim - 1,
        np.array(facet_indices, dtype=np.int32),
        np.arange(len(facet_indices), dtype=np.int32),
    )

    msh1, parent_cell, parent_facet = refine_plaza_wrapper(msh)

    msh1.topology.create_entities(tdim - 1)
    new_meshtag = transfer_meshtag(meshtag, msh1, parent_cell, parent_facet)
    assert len(new_meshtag.indices) == (tdim * 2 - 2) * len(meshtag.indices)

    # New tags should be on facets with one cell (i.e. exterior)
    msh1.topology.create_connectivity(tdim - 1, tdim)
    new_f_to_c = msh1.topology.connectivity(tdim - 1, tdim)
    for f in new_meshtag.indices:
        assert len(new_f_to_c.links(f)) == 1

    # Now mark all facets (including internal)
    facet_indices = np.arange(msh.topology.index_map(tdim - 1).size_local)
    meshtag = meshtags(
        msh,
        tdim - 1,
        np.array(facet_indices, dtype=np.int32),
        np.arange(len(facet_indices), dtype=np.int32),
    )
    new_meshtag = transfer_meshtag(meshtag, msh1, parent_cell, parent_facet)
    assert len(new_meshtag.indices) == (tdim * 2 - 2) * len(meshtag.indices)


@pytest.mark.parametrize("tdim", [2, 3])
@pytest.mark.parametrize(
    "refine_plaza_wrapper",
    [
        lambda msh: refine(msh, option=RefinementOption.parent_cell_and_facet),
        lambda msh: refine(
            msh,
            np.arange(msh.topology.index_map(1).size_local),
            option=RefinementOption.parent_cell_and_facet,
        ),
    ],
)
def test_refine_cell_meshtag(tdim, refine_plaza_wrapper):
    if tdim == 3:
        msh = create_unit_cube(
            MPI.COMM_WORLD, 2, 3, 5, CellType.tetrahedron, ghost_mode=GhostMode.none
        )
    else:
        msh = create_unit_square(MPI.COMM_WORLD, 2, 5, CellType.triangle, ghost_mode=GhostMode.none)

    msh.topology.create_entities(1)
    cell_indices = np.arange(msh.topology.index_map(tdim).size_local)
    meshtag = meshtags(
        msh,
        tdim,
        np.array(cell_indices, dtype=np.int32),
        np.arange(len(cell_indices), dtype=np.int32),
    )

    msh1, parent_cell, _ = refine_plaza_wrapper(msh)
    new_meshtag = transfer_meshtag(meshtag, msh1, parent_cell)
    assert sum(new_meshtag.values) == (tdim * 4 - 4) * sum(meshtag.values)
    assert len(new_meshtag.indices) == (tdim * 4 - 4) * len(meshtag.indices)


def test_refine_ufl_cargo():
    msh = create_unit_cube(MPI.COMM_WORLD, 4, 3, 3)
    msh.topology.create_entities(1)
    msh1, _, _ = refine(msh)
    assert msh1.ufl_domain().ufl_cargo() != msh.ufl_domain().ufl_cargo()
