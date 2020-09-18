# Copyright (C) 2019 Chris Richardson
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly over domains"""

import dolfinx
import numpy
import pytest
import ufl
from dolfinx.mesh import locate_entities_boundary
from mpi4py import MPI
from petsc4py import PETSc


@pytest.fixture
def mesh():
    return dolfinx.generation.UnitSquareMesh(MPI.COMM_WORLD, 10, 10)


parametrize_ghost_mode = pytest.mark.parametrize("mode", [
    pytest.param(dolfinx.cpp.mesh.GhostMode.none,
                 marks=pytest.mark.skipif(condition=MPI.COMM_WORLD.size > 1,
                                          reason="Unghosted interior facets fail in parallel")),
    pytest.param(dolfinx.cpp.mesh.GhostMode.shared_facet,
                 marks=pytest.mark.skipif(condition=MPI.COMM_WORLD.size == 1,
                                          reason="Shared ghost modes fail in serial"))])


@pytest.mark.parametrize("mode", [dolfinx.cpp.mesh.GhostMode.none, dolfinx.cpp.mesh.GhostMode.shared_facet])
def test_assembly_dx_domains(mode):
    mesh = dolfinx.generation.UnitSquareMesh(MPI.COMM_WORLD, 10, 10, ghost_mode=mode)
    V = dolfinx.FunctionSpace(mesh, ("CG", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    # Prepare a marking structures
    # indices cover all cells
    # values are [1, 2, 3, 3, ...]
    cell_map = mesh.topology.index_map(mesh.topology.dim)
    num_cells = cell_map.size_local + cell_map.num_ghosts
    indices = numpy.arange(0, num_cells)
    values = numpy.full(indices.shape, 3, dtype=numpy.intc)
    values[0] = 1
    values[1] = 2
    marker = dolfinx.mesh.MeshTags(mesh, mesh.topology.dim, indices, values)
    dx = ufl.Measure('dx', subdomain_data=marker, domain=mesh)
    w = dolfinx.Function(V)
    with w.vector.localForm() as w_local:
        w_local.set(0.5)

    # Assemble matrix
    a = w * ufl.inner(u, v) * (dx(1) + dx(2) + dx(3))
    A = dolfinx.fem.assemble_matrix(a)
    A.assemble()
    a2 = w * ufl.inner(u, v) * dx
    A2 = dolfinx.fem.assemble_matrix(a2)
    A2.assemble()
    assert (A - A2).norm() < 1.0e-12

    # Assemble vector
    L = ufl.inner(w, v) * (dx(1) + dx(2) + dx(3))
    b = dolfinx.fem.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    L2 = ufl.inner(w, v) * dx
    b2 = dolfinx.fem.assemble_vector(L2)
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    print(b2.norm())
    assert (b - b2).norm() < 1.0e-12

    # Assemble scalar
    L = w * (dx(1) + dx(2) + dx(3))
    s = dolfinx.fem.assemble_scalar(L)
    s = mesh.mpi_comm().allreduce(s, op=MPI.SUM)
    assert s == pytest.approx(0.5, 1.0e-12)
    L2 = w * dx
    s2 = dolfinx.fem.assemble_scalar(L2)
    s2 = mesh.mpi_comm().allreduce(s2, op=MPI.SUM)
    assert s == pytest.approx(s2, 1.0e-12)


@pytest.mark.parametrize("mode", [dolfinx.cpp.mesh.GhostMode.none, dolfinx.cpp.mesh.GhostMode.shared_facet])
def test_assembly_ds_domains(mode):
    mesh = dolfinx.generation.UnitSquareMesh(MPI.COMM_WORLD, 10, 10, ghost_mode=mode)
    V = dolfinx.FunctionSpace(mesh, ("CG", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    def bottom(x):
        return numpy.isclose(x[1], 0.0)

    def top(x):
        return numpy.isclose(x[1], 1.0)

    def left(x):
        return numpy.isclose(x[0], 0.0)

    def right(x):
        return numpy.isclose(x[0], 1.0)

    bottom_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, bottom)
    bottom_vals = numpy.full(bottom_facets.shape, 1, numpy.intc)

    top_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, top)
    top_vals = numpy.full(top_facets.shape, 2, numpy.intc)

    left_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, left)
    left_vals = numpy.full(left_facets.shape, 3, numpy.intc)

    right_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, right)
    right_vals = numpy.full(right_facets.shape, 6, numpy.intc)

    indices = numpy.hstack((bottom_facets, top_facets, left_facets, right_facets))
    values = numpy.hstack((bottom_vals, top_vals, left_vals, right_vals))

    indices, pos = numpy.unique(indices, return_index=True)
    marker = dolfinx.mesh.MeshTags(mesh, mesh.topology.dim - 1, indices, values[pos])

    ds = ufl.Measure('ds', subdomain_data=marker, domain=mesh)

    w = dolfinx.Function(V)
    with w.vector.localForm() as w_local:
        w_local.set(0.5)

    # Assemble matrix
    a = w * ufl.inner(u, v) * (ds(1) + ds(2) + ds(3) + ds(6))
    A = dolfinx.fem.assemble_matrix(a)
    A.assemble()
    norm1 = A.norm()
    a2 = w * ufl.inner(u, v) * ds
    A2 = dolfinx.fem.assemble_matrix(a2)
    A2.assemble()
    norm2 = A2.norm()
    assert norm1 == pytest.approx(norm2, 1.0e-12)

    # Assemble vector
    L = ufl.inner(w, v) * (ds(1) + ds(2) + ds(3) + ds(6))
    b = dolfinx.fem.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    L2 = ufl.inner(w, v) * ds
    b2 = dolfinx.fem.assemble_vector(L2)
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert b.norm() == pytest.approx(b2.norm(), 1.0e-12)

    # Assemble scalar
    L = w * (ds(1) + ds(2) + ds(3) + ds(6))
    s = dolfinx.fem.assemble_scalar(L)
    s = mesh.mpi_comm().allreduce(s, op=MPI.SUM)
    L2 = w * ds
    s2 = dolfinx.fem.assemble_scalar(L2)
    s2 = mesh.mpi_comm().allreduce(s2, op=MPI.SUM)
    assert (s == pytest.approx(s2, 1.0e-12) and 2.0 == pytest.approx(s, 1.0e-12))


@parametrize_ghost_mode
def test_assembly_dS_domains(mode):
    N = 10
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, N, N, ghost_mode=mode)
    one = dolfinx.Constant(mesh, 1)
    val = dolfinx.fem.assemble_scalar(one * ufl.dS)
    val = mesh.mpi_comm().allreduce(val, op=MPI.SUM)
    assert val == pytest.approx(2 * (N - 1) + N * numpy.sqrt(2), 1.0e-7)


@parametrize_ghost_mode
def test_additivity(mode):
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 12, 12, ghost_mode=mode)
    V = dolfinx.FunctionSpace(mesh, ("CG", 1))

    f1 = dolfinx.Function(V)
    f2 = dolfinx.Function(V)
    f3 = dolfinx.Function(V)
    with f1.vector.localForm() as f1_local:
        f1_local.set(1.0)
    with f2.vector.localForm() as f2_local:
        f2_local.set(2.0)
    with f3.vector.localForm() as f3_local:
        f3_local.set(3.0)
    j1 = ufl.inner(f1, f1) * ufl.dx(mesh)
    j2 = ufl.inner(f2, f2) * ufl.ds(mesh)
    j3 = ufl.inner(ufl.avg(f3), ufl.avg(f3)) * ufl.dS(mesh)

    # Assemble each scalar form separately
    J1 = mesh.mpi_comm().allreduce(dolfinx.fem.assemble_scalar(j1), op=MPI.SUM)
    J2 = mesh.mpi_comm().allreduce(dolfinx.fem.assemble_scalar(j2), op=MPI.SUM)
    J3 = mesh.mpi_comm().allreduce(dolfinx.fem.assemble_scalar(j3), op=MPI.SUM)

    # Sum forms and assemble the result
    J12 = mesh.mpi_comm().allreduce(dolfinx.fem.assemble_scalar(j1 + j2), op=MPI.SUM)
    J13 = mesh.mpi_comm().allreduce(dolfinx.fem.assemble_scalar(j1 + j3), op=MPI.SUM)
    J23 = mesh.mpi_comm().allreduce(dolfinx.fem.assemble_scalar(j2 + j3), op=MPI.SUM)
    J123 = mesh.mpi_comm().allreduce(dolfinx.fem.assemble_scalar(j1 + j2 + j3), op=MPI.SUM)

    # Compare assembled values
    assert (J1 + J2) == pytest.approx(J12)
    assert (J1 + J3) == pytest.approx(J13)
    assert (J2 + J3) == pytest.approx(J23)
    assert (J1 + J2 + J3) == pytest.approx(J123)
