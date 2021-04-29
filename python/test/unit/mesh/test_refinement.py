# Copyright (C) 2018 Chris N Richardson
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx
import numpy as np
import pytest
import ufl
from dolfinx_utils.test.skips import skip_in_parallel
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
    assert Q


def test_RefineUnitCubeMesh_keep_partition():
    """Refine mesh of unit cube."""
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 5, 7, 9, ghost_mode=GhostMode.none)
    mesh.topology.create_entities(1)
    mesh = refine(mesh, redistribute=False)
    assert mesh.topology.index_map(0).size_global == 3135
    assert mesh.topology.index_map(3).size_global == 15120
    Q = FunctionSpace(mesh, ("CG", 1))
    assert Q


def test_refine_create_form():
    """Check that forms can be assembled on refined mesh"""
    mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, 3, 3, 3)
    mesh.topology.create_entities(1)
    mesh = refine(mesh, redistribute=True)

    V = dolfinx.FunctionSpace(mesh, ("CG", 1))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    dolfinx.fem.assemble_matrix(a)


def xtest_refinement_gdim():
    """Test that 2D refinement is still 2D"""
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 3, 4, ghost_mode=GhostMode.none)
    mesh2 = refine(mesh, redistribute=True)
    assert mesh.geometry.dim == mesh2.geometry.dim


@skip_in_parallel
@pytest.mark.parametrize("tdim", [2, 3])
@pytest.mark.parametrize("mesh_size", [2, 4])
def test_parent_map(tdim, mesh_size):
    if tdim == 2:
        mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, mesh_size, mesh_size)
    elif tdim == 3:
        mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, mesh_size, mesh_size, mesh_size)

    mesh.topology.create_entities(1)
    mesh_refined = dolfinx.cpp.refinement.refine(mesh)

    x = mesh.geometry.x
    x_refined = mesh_refined.geometry.x

    t_to_g = {}
    con = mesh.topology.connectivity(tdim, 0)
    for cell in range(con.num_nodes):
        tdofs = con.links(cell)
        gdofs = mesh.geometry.dofmap.links(cell)
        for i, j in zip(tdofs, gdofs):
            t_to_g[i] = j
    t_to_g_refined = {}
    con = mesh_refined.topology.connectivity(tdim, 0)
    for cell in range(con.num_nodes):
        tdofs = con.links(cell)
        gdofs = mesh_refined.geometry.dofmap.links(cell)
        for i, j in zip(tdofs, gdofs):
            t_to_g_refined[i] = j

    e_to_v = mesh.topology.connectivity(1, 0)

    for i, (dim, n) in mesh_refined.parent_map.items():
        if dim == 0:
            assert np.allclose(x_refined[t_to_g_refined[i]], x[t_to_g[n]])
        if dim == 1:
            e0, e1 = e_to_v.links(n)
            midpoint = (x[t_to_g[e0]] + x[t_to_g[e1]]) / 2
            assert np.allclose(x_refined[t_to_g_refined[i]], midpoint)
