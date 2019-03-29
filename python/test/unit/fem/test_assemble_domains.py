# Copyright (C) 2019 Chris Richardson
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly over domains"""

import numpy
import pytest

import dolfin
import ufl


@pytest.fixture
def mesh():
    return dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 10, 10)


def test_assembly_dx_domains(mesh):
    V = dolfin.FunctionSpace(mesh, ("CG", 1))
    u, v = dolfin.TrialFunction(V), dolfin.TestFunction(V)

    marker = dolfin.MeshFunction("size_t", mesh, mesh.topology.dim, 0)

    def bottom_fifth(x):
        return numpy.less_equal(x[:, 1], 0.2)

    def upper_fifth(x):
        return numpy.greater_equal(x[:, 1], 0.8)

    bottom_fifth_subdomain = dolfin.mesh.create_subdomain(bottom_fifth)
    bottom_fifth_subdomain.mark(marker, 111)

    upper_fifth_subdomain = dolfin.mesh.create_subdomain(upper_fifth)
    upper_fifth_subdomain.mark(marker, 222)

    dx = dolfin.Measure('dx', subdomain_data=marker, domain=mesh)

    #
    # Assemble matrix
    #

    # Integration just over marked cells
    a = ufl.inner(u, v) * dx(111) + ufl.inner(u, v) * dx(222)

    A = dolfin.fem.assemble_matrix(a)
    A.assemble()
    norm1 = A.norm()

    # Integration over whole domain, but effectively zeroing
    # with contitional
    x = ufl.SpatialCoordinate(mesh)
    conditional_marker = ufl.conditional(ufl.Or(x[0] <= 0.2, x[0] >= 0.8), 1.0, 0.0)

    a2 = conditional_marker * ufl.inner(u, v) * dx

    A2 = dolfin.fem.assemble_matrix(a2)
    A2.assemble()
    norm2 = A2.norm()

    assert norm1 == pytest.approx(norm2, 1.0e-12)

    #
    # Assemble vector
    #

    L = v * dx(111) + v * dx(222)
    b = dolfin.fem.assemble_vector(L)

    L2 = conditional_marker * v * dx
    b2 = dolfin.fem.assemble_vector(L2)

    assert b.norm() == pytest.approx(b2.norm(), 1.0e-12)

    #
    # Assemble scalar
    #

    L = 1.0 * dx(111) + 1.0 * dx(222)
    s = dolfin.fem.assemble_scalar(L)
    s = dolfin.MPI.sum(mesh.mpi_comm(), s)

    L2 = conditional_marker * 1.0 * dx
    s2 = dolfin.fem.assemble_scalar(L2)
    s2 = dolfin.MPI.sum(mesh.mpi_comm(), s2)

    assert (s == pytest.approx(s2, 1.0e-12) and 0.4 == pytest.approx(s, 1.0e-12))


def test_assembly_ds_domains(mesh):
    V = dolfin.FunctionSpace(mesh, ("CG", 1))
    u, v = dolfin.TrialFunction(V), dolfin.TestFunction(V)

    marker = dolfin.MeshFunction("size_t", mesh, mesh.topology.dim - 1, 0)

    def bottom(x):
        return numpy.isclose(x[:, 1], 0.0)

    def top(x):
        return numpy.isclose(x[:, 1], 1.0)

    bottom_subdomain = dolfin.mesh.create_subdomain(bottom)
    bottom_subdomain.mark(marker, 111)

    top_subdomain = dolfin.mesh.create_subdomain(top)
    top_subdomain.mark(marker, 222)

    ds = dolfin.Measure('ds', subdomain_data=marker, domain=mesh)

    #
    # Assemble matrix
    #

    # Integration just over marked cells
    a = ufl.inner(u, v) * ds(111) + ufl.inner(u, v) * ds(222)

    A = dolfin.fem.assemble_matrix(a)
    A.assemble()
    norm1 = A.norm()

    # Integration over whole domain, but effectively zeroing
    # with contitional
    x = ufl.SpatialCoordinate(mesh)
    conditional_marker = ufl.conditional(ufl.Or(ufl.eq(x[0], 0.0), ufl.eq(x[0], 1.0)), 1.0, 0.0)

    a2 = conditional_marker * ufl.inner(u, v) * ds

    A2 = dolfin.fem.assemble_matrix(a2)
    A2.assemble()
    norm2 = A2.norm()

    assert norm1 == pytest.approx(norm2, 1.0e-12)

    # L = 1.0 * ds(1)
    # s = dolfin.fem.assemble_scalar(L)
    # s = dolfin.MPI.sum(mesh.mpi_comm(), s)

    # assert 1.0 == pytest.approx(1.0, 1.0e-12)
