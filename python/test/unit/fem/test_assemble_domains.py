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

    def bottom_half(x):
        return numpy.less_equal(x[:, 1], 0.2)

    bottom_half_subdomain = dolfin.mesh.create_subdomain(bottom_half)
    bottom_half_subdomain.mark(marker, 111)

    dx = dolfin.Measure('dx', subdomain_data=marker)

    #
    # Assemble matrix
    #

    # Integration just over marked cells
    a = ufl.inner(u, v) * dx(111)

    A = dolfin.fem.assemble_matrix(a)
    A.assemble()
    norm1 = A.norm()

    # Integration over whole domain, but effectively zeroing
    # with contitional
    x = ufl.SpatialCoordinate(mesh)
    conditional_marker = ufl.conditional(x[0] <= 0.2, 1.0, 0.0)

    a2 = conditional_marker * ufl.inner(u, v) * dx

    A2 = dolfin.fem.assemble_matrix(a2)
    A2.assemble()
    norm2 = A2.norm()

    assert numpy.isclose(norm1, norm2)

    #
    # Assemble vector
    #

    L = v * dx(111)
    b = dolfin.fem.assemble_vector(L)

    L2 = conditional_marker * v * dx
    b2 = dolfin.fem.assemble_vector(L2)

    assert numpy.isclose(b.norm(), b2.norm())

    #
    # Assemble scalar
    #

    L = 1.0 * dx(111, domain=mesh)
    s = dolfin.fem.assemble_scalar(L)
    s = dolfin.MPI.sum(mesh.mpi_comm(), s)

    L2 = conditional_marker * 1.0 * dx(domain=mesh)
    s2 = dolfin.fem.assemble_scalar(L2)
    s2 = dolfin.MPI.sum(mesh.mpi_comm(), s2)

    print(s, s2)
    assert numpy.isclose(s, s2) and numpy.isclose(s, 0.5)


def test_assembly_ds_domains(mesh):
    V = dolfin.FunctionSpace(mesh, ("CG", 1))
    u, v = dolfin.TrialFunction(V), dolfin.TestFunction(V)

    marker = dolfin.MeshFunction("size_t", mesh, mesh.topology.dim - 1, 0)

    def bottom(x):
        return numpy.isclose(x[:, 1], 10.0)

    bottom_subdomain = dolfin.mesh.create_subdomain(bottom)
    bottom_subdomain.mark(marker, 1)

    ds = dolfin.Measure('ds', subdomain_data=marker, domain=mesh)

    L = 1.0 * ds(1)
    s = dolfin.fem.assemble_scalar(L)
    print(s)
