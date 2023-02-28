# Copyright (C) 2019 Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly over domains"""

import numpy as np
import pytest

import ufl
from dolfinx import cpp as _cpp
from dolfinx.fem import (Constant, Function, FunctionSpace, assemble_scalar,
                         dirichletbc, form)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               set_bc)
from dolfinx.mesh import (GhostMode, Mesh, create_unit_square,
                          locate_entities_boundary, meshtags,
                          meshtags_from_entities)

from mpi4py import MPI
from petsc4py import PETSc


@pytest.fixture
def mesh():
    return create_unit_square(MPI.COMM_WORLD, 10, 10)


def create_cell_meshtags_from_entities(mesh: Mesh, dim: int, cells: np.ndarray, values: np.ndarray):
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    cell_to_vertices = mesh.topology.connectivity(mesh.topology.dim, 0)
    entities = _cpp.graph.AdjacencyList_int32([cell_to_vertices.links(cell) for cell in cells])
    return meshtags_from_entities(mesh, dim, entities, values)


parametrize_ghost_mode = pytest.mark.parametrize("mode", [
    pytest.param(GhostMode.none, marks=pytest.mark.skipif(condition=MPI.COMM_WORLD.size > 1,
                                                          reason="Unghosted interior facets fail in parallel")),
    pytest.param(GhostMode.shared_facet, marks=pytest.mark.skipif(condition=MPI.COMM_WORLD.size == 1,
                                                                  reason="Shared ghost modes fail in serial"))])


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
@pytest.mark.parametrize("meshtags_factory", [meshtags, create_cell_meshtags_from_entities])
def test_assembly_dx_domains(mode, meshtags_factory):
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10, ghost_mode=mode)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    # Prepare a marking structures
    # indices cover all cells
    # values are [1, 2, 3, 3, ...]
    cell_map = mesh.topology.index_map(mesh.topology.dim)
    num_cells = cell_map.size_local + cell_map.num_ghosts
    indices = np.arange(0, num_cells)
    values = np.full(indices.shape, 3, dtype=np.intc)
    values[0] = 1
    values[1] = 2
    marker = meshtags_factory(mesh, mesh.topology.dim, indices, values)
    dx = ufl.Measure('dx', subdomain_data=marker, domain=mesh)
    w = Function(V)
    w.x.array[:] = 0.5

    # Assemble matrix
    a = form(w * ufl.inner(u, v) * (dx(1) + dx(2) + dx(3)))
    A = assemble_matrix(a)
    A.assemble()
    a2 = form(w * ufl.inner(u, v) * dx)
    A2 = assemble_matrix(a2)
    A2.assemble()
    assert (A - A2).norm() < 1.0e-12

    bc = dirichletbc(Function(V), range(30))

    # Assemble vector
    L = form(ufl.inner(w, v) * (dx(1) + dx(2) + dx(3)))
    b = assemble_vector(L)

    apply_lifting(b, [a], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    L2 = form(ufl.inner(w, v) * dx)
    b2 = assemble_vector(L2)
    apply_lifting(b2, [a], [[bc]])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                   mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, [bc])
    assert (b - b2).norm() < 1.0e-12

    # Assemble scalar
    L = form(w * (dx(1) + dx(2) + dx(3)))
    s = assemble_scalar(L)
    s = mesh.comm.allreduce(s, op=MPI.SUM)
    assert s == pytest.approx(0.5, 1.0e-12)
    L2 = form(w * dx)
    s2 = assemble_scalar(L2)
    s2 = mesh.comm.allreduce(s2, op=MPI.SUM)
    assert s == pytest.approx(s2, 1.0e-12)

    A.destroy()
    b.destroy()
    A2.destroy()
    b2.destroy()


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
def test_assembly_ds_domains(mode):
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10, ghost_mode=mode)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    def bottom(x):
        return np.isclose(x[1], 0.0)

    def top(x):
        return np.isclose(x[1], 1.0)

    def left(x):
        return np.isclose(x[0], 0.0)

    def right(x):
        return np.isclose(x[0], 1.0)

    bottom_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, bottom)
    bottom_vals = np.full(bottom_facets.shape, 1, np.intc)

    top_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, top)
    top_vals = np.full(top_facets.shape, 2, np.intc)

    left_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, left)
    left_vals = np.full(left_facets.shape, 3, np.intc)

    right_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, right)
    right_vals = np.full(right_facets.shape, 6, np.intc)

    indices = np.hstack((bottom_facets, top_facets, left_facets, right_facets))
    values = np.hstack((bottom_vals, top_vals, left_vals, right_vals))

    indices, pos = np.unique(indices, return_index=True)
    marker = meshtags(mesh, mesh.topology.dim - 1, indices, values[pos])

    ds = ufl.Measure('ds', subdomain_data=marker, domain=mesh)

    w = Function(V)
    w.x.array[:] = 0.5

    bc = dirichletbc(Function(V), range(30))

    # Assemble matrix
    a = form(w * ufl.inner(u, v) * (ds(1) + ds(2) + ds(3) + ds(6)))
    A = assemble_matrix(a)
    A.assemble()
    norm1 = A.norm()
    a2 = form(w * ufl.inner(u, v) * ds)
    A2 = assemble_matrix(a2)
    A2.assemble()
    norm2 = A2.norm()
    assert norm1 == pytest.approx(norm2, 1.0e-12)

    # Assemble vector
    L = form(ufl.inner(w, v) * (ds(1) + ds(2) + ds(3) + ds(6)))
    b = assemble_vector(L)

    apply_lifting(b, [a], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    L2 = form(ufl.inner(w, v) * ds)
    b2 = assemble_vector(L2)
    apply_lifting(b2, [a2], [[bc]])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, [bc])

    assert b.norm() == pytest.approx(b2.norm(), 1.0e-12)

    # Assemble scalar
    L = form(w * (ds(1) + ds(2) + ds(3) + ds(6)))
    s = assemble_scalar(L)
    s = mesh.comm.allreduce(s, op=MPI.SUM)
    L2 = form(w * ds)
    s2 = assemble_scalar(L2)
    s2 = mesh.comm.allreduce(s2, op=MPI.SUM)
    assert (s == pytest.approx(s2, 1.0e-12) and 2.0 == pytest.approx(s, 1.0e-12))

    A.destroy()
    b.destroy()
    A2.destroy()
    b2.destroy()


@parametrize_ghost_mode
def test_assembly_dS_domains(mode):
    N = 10
    mesh = create_unit_square(MPI.COMM_WORLD, N, N, ghost_mode=mode)
    one = Constant(mesh, PETSc.ScalarType(1))
    val = assemble_scalar(form(one * ufl.dS))
    val = mesh.comm.allreduce(val, op=MPI.SUM)
    assert val == pytest.approx(2 * (N - 1) + N * np.sqrt(2), 1.0e-7)


@parametrize_ghost_mode
def test_additivity(mode):
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode)
    V = FunctionSpace(mesh, ("Lagrange", 1))

    f1 = Function(V)
    f2 = Function(V)
    f3 = Function(V)
    f1.x.array[:] = 1.0
    f2.x.array[:] = 2.0
    f3.x.array[:] = 3.0
    j1 = ufl.inner(f1, f1) * ufl.dx(mesh)
    j2 = ufl.inner(f2, f2) * ufl.ds(mesh)
    j3 = ufl.inner(ufl.avg(f3), ufl.avg(f3)) * ufl.dS(mesh)

    # Assemble each scalar form separately
    J1 = mesh.comm.allreduce(assemble_scalar(form(j1)), op=MPI.SUM)
    J2 = mesh.comm.allreduce(assemble_scalar(form(j2)), op=MPI.SUM)
    J3 = mesh.comm.allreduce(assemble_scalar(form(j3)), op=MPI.SUM)

    # Sum forms and assemble the result
    J12 = mesh.comm.allreduce(assemble_scalar(form(j1 + j2)), op=MPI.SUM)
    J13 = mesh.comm.allreduce(assemble_scalar(form(j1 + j3)), op=MPI.SUM)
    J23 = mesh.comm.allreduce(assemble_scalar(form(j2 + j3)), op=MPI.SUM)
    J123 = mesh.comm.allreduce(assemble_scalar(form(j1 + j2 + j3)), op=MPI.SUM)

    # Compare assembled values
    assert (J1 + J2) == pytest.approx(J12)
    assert (J1 + J3) == pytest.approx(J13)
    assert (J2 + J3) == pytest.approx(J23)
    assert (J1 + J2 + J3) == pytest.approx(J123)
