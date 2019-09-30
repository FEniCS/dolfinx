# Copyright (C) 2019 Chris Richardson
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly over domains"""

import numpy
import pytest
from petsc4py import PETSc

import dolfin
import ufl


@pytest.fixture
def mesh():
    return dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 10, 10)


def test_assembly_dx_domains(mesh):
    V = dolfin.FunctionSpace(mesh, ("CG", 1))
    u, v = dolfin.TrialFunction(V), dolfin.TestFunction(V)

    marker = dolfin.MeshFunction("size_t", mesh, mesh.topology.dim, 0)
    values = marker.values
    # Mark first, second and all other
    # Their union is the whole domain
    values[0] = 111
    values[1] = 222
    values[2:] = 333

    dx = ufl.Measure('dx', subdomain_data=marker, domain=mesh)

    w = dolfin.Function(V)
    with w.vector.localForm() as w_local:
        w_local.set(0.5)

    #
    # Assemble matrix
    #

    a = w * ufl.inner(u, v) * (dx(111) + dx(222) + dx(333))

    A = dolfin.fem.assemble_matrix(a)
    A.assemble()
    norm1 = A.norm()

    a2 = w * ufl.inner(u, v) * dx

    A2 = dolfin.fem.assemble_matrix(a2)
    A2.assemble()
    norm2 = A2.norm()

    assert norm1 == pytest.approx(norm2, 1.0e-12)

    #
    # Assemble vector
    #

    L = ufl.inner(w, v) * (dx(111) + dx(222) + dx(333))
    b = dolfin.fem.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    L2 = ufl.inner(w, v) * dx
    b2 = dolfin.fem.assemble_vector(L2)
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    assert b.norm() == pytest.approx(b2.norm(), 1.0e-12)

    #
    # Assemble scalar
    #

    L = w * (dx(111) + dx(222) + dx(333))
    s = dolfin.fem.assemble_scalar(L)
    s = dolfin.MPI.sum(mesh.mpi_comm(), s)

    L2 = w * dx
    s2 = dolfin.fem.assemble_scalar(L2)
    s2 = dolfin.MPI.sum(mesh.mpi_comm(), s2)

    assert (s == pytest.approx(s2, 1.0e-12) and 0.5 == pytest.approx(s, 1.0e-12))


def test_assembly_ds_domains(mesh):
    V = dolfin.FunctionSpace(mesh, ("CG", 1))
    u, v = dolfin.TrialFunction(V), dolfin.TestFunction(V)

    marker = dolfin.MeshFunction("size_t", mesh, mesh.topology.dim - 1, 0)

    def bottom(x):
        return numpy.isclose(x[:, 1], 0.0)

    def top(x):
        return numpy.isclose(x[:, 1], 1.0)

    def left(x):
        return numpy.isclose(x[:, 0], 0.0)

    def right(x):
        return numpy.isclose(x[:, 0], 1.0)

    marker.mark(bottom, 111)
    marker.mark(top, 222)
    marker.mark(left, 333)
    marker.mark(right, 444)

    ds = ufl.Measure('ds', subdomain_data=marker, domain=mesh)

    w = dolfin.Function(V)
    with w.vector.localForm() as w_local:
        w_local.set(0.5)

    #
    # Assemble matrix
    #

    a = w * ufl.inner(u, v) * (ds(111) + ds(222) + ds(333) + ds(444))

    A = dolfin.fem.assemble_matrix(a)
    A.assemble()
    norm1 = A.norm()

    a2 = w * ufl.inner(u, v) * ds

    A2 = dolfin.fem.assemble_matrix(a2)
    A2.assemble()
    norm2 = A2.norm()

    assert norm1 == pytest.approx(norm2, 1.0e-12)

    #
    # Assemble vector
    #

    L = ufl.inner(w, v) * (ds(111) + ds(222) + ds(333) + ds(444))
    b = dolfin.fem.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    L2 = ufl.inner(w, v) * ds
    b2 = dolfin.fem.assemble_vector(L2)
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    assert b.norm() == pytest.approx(b2.norm(), 1.0e-12)

    #
    # Assemble scalar
    #

    L = w * (ds(111) + ds(222) + ds(333) + ds(444))
    s = dolfin.fem.assemble_scalar(L)
    s = dolfin.MPI.sum(mesh.mpi_comm(), s)

    L2 = w * ds
    s2 = dolfin.fem.assemble_scalar(L2)
    s2 = dolfin.MPI.sum(mesh.mpi_comm(), s2)

    assert (s == pytest.approx(s2, 1.0e-12) and 2.0 == pytest.approx(s, 1.0e-12))


@pytest.mark.parametrize("mode",
                         [pytest.param(dolfin.cpp.mesh.GhostMode.none,
                                       marks=pytest.mark.skipif(condition=dolfin.MPI.size(dolfin.MPI.comm_world) > 1,
                                                                reason="Unghosted interior facets fail in parallel")),
                          pytest.param(dolfin.cpp.mesh.GhostMode.shared_facet,
                                       marks=pytest.mark.skipif(condition=dolfin.MPI.size(dolfin.MPI.comm_world) == 1,
                                                                reason="Shared ghost modes fail in serial")),
                          pytest.param(dolfin.cpp.mesh.GhostMode.shared_vertex,
                                       marks=pytest.mark.skipif(condition=dolfin.MPI.size(dolfin.MPI.comm_world) == 1,
                                                                reason="Shared ghost modes fail in serial"))])
def test_assembly_dS_domains(mode):
    N = 10
    mesh = dolfin.UnitSquareMesh(dolfin.MPI.comm_world, N, N, ghost_mode=mode)
    one = dolfin.Constant(mesh, 1)
    val = dolfin.fem.assemble_scalar(one * ufl.dS)
    val = dolfin.MPI.sum(mesh.mpi_comm(), val)
    assert val == pytest.approx(2 * (N - 1) + N * numpy.sqrt(2), 1.0e-7)
