"""Unit tests for Dirichlet boundary conditions"""

# Copyright (C) 2011-2017 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import pytest
import numpy

from dolfin import *
from dolfin_utils.test import skip_in_parallel, datadir


def test_instantiation():
    """ A rudimentary test for instantiation"""
    # FIXME: Needs to be expanded
    mesh = UnitCubeMesh(MPI.comm_world, 8, 8, 8)
    V = FunctionSpace(mesh, "CG", 1)

    bc0 = DirichletBC(V, 1, "x[0]<0")
    bc1 = DirichletBC(bc0)
    assert bc0.function_space() == bc1.function_space()


def test_director_lifetime():
    """Test for any problems with objects with directors going out
    of scope"""

    class Boundary(SubDomain):
        def inside(self, x, on_boundary): return on_boundary

    class BoundaryFunction(UserExpression):
        def eval(self, values, x): values[0] = 1.0

    mesh = UnitSquareMesh(MPI.comm_world, 8, 8)
    V = FunctionSpace(mesh, "Lagrange", 1)
    v, u = TestFunction(V), TrialFunction(V)

    A0 = assemble(v*u*dx)
    bc0 = DirichletBC(V, BoundaryFunction(degree=1), Boundary())
    bc0.apply(A0)

    bc1 = DirichletBC(V, Expression("1.0", degree=0), CompiledSubDomain("on_boundary"))
    A1 = assemble(v*u*dx)
    bc1.apply(A1)

    assert round(A1.norm("frobenius") - A0.norm("frobenius"), 7) == 0


def xtest_get_values():
    mesh = UnitSquareMesh(MPI.comm_world, 8, 8)
    dofs = numpy.zeros(3, dtype="I")

    def upper(x, on_boundary):
        return x[1] > 0.5 + DOLFIN_EPS

    V = FunctionSpace(mesh, "CG", 1)
    bc = DirichletBC(V, 0.0, upper)
    bc_values = bc.get_boundary_values()


def test_user_meshfunction_domains():
    mesh0 = UnitSquareMesh(MPI.comm_world, 12, 12)
    mesh1 = UnitSquareMesh(MPI.comm_world, 12, 12)
    V = FunctionSpace(mesh0, "CG", 1)

    DirichletBC(V, Constant(0.0), MeshFunction("size_t", mesh0, 1), 0)
    DirichletBC(V, Constant(0.0), MeshFunction("size_t", mesh0, mesh0.topology.dim-1), 0)
    with pytest.raises(RuntimeError):
        DirichletBC(V, 0.0, MeshFunction("size_t", mesh0, mesh0.topology.dim), 0)
        DirichletBC(V, 0.0, MeshFunction("size_t", mesh0, 0), 0)
        DirichletBC(V, 0.0, MeshFunction("size_t", mesh1, mesh1.topology.dim-1), 0)


@skip_in_parallel
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("element_type",
                         ["RT", "DRT", "BDM", "N1curl", "N2curl"])
def test_bc_for_piola_on_manifolds(element_type, degree):
    """Testing DirichletBC for piolas over standard domains vs manifolds.

    """
    n = 4
    side = CompiledSubDomain("near(x[2], 0.0)")
    mesh = SubMesh(MPI.comm_world, BoundaryMesh(MPI.comm_world, UnitCubeMesh(MPI.comm_world, n, n, n), "exterior"), side)
    mesh.init_cell_orientations(Expression(("0.0", "0.0", "1.0"), degree=0))
    square = UnitSquareMesh(MPI.comm_world, n, n)

    V = FunctionSpace(mesh, element_type, degree)
    bc = DirichletBC(V, (1.0, 0.0, 0.0), "on_boundary")
    u = Function(V)
    bc.apply(u.vector())
    b0 = assemble(inner(u, u)*dx)

    V = FunctionSpace(square, element_type, degree)
    bc = DirichletBC(V, (1.0, 0.0), "on_boundary")
    u = Function(V)
    bc.apply(u.vector())
    b1 = assemble(inner(u, u)*dx)
    assert round(b0 - b1, 7) == 0


def test_zero():
    mesh = UnitSquareMesh(MPI.comm_world, 4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    u1 = interpolate(Constant(1.0), V)

    bc = DirichletBC(V, 0, "on_boundary")

    # Create arbitrary matrix of size V.dim
    #
    # Note: Identity matrix would suffice, but there doesn't seem
    # an easy way to construct it in dolfin

    v, u = TestFunction(V), TrialFunction(V)
    A0 = assemble(u*v*dx)

    # Zero rows at boundary dofs
    bc.zero(A0)

    u1_zero = Function(V)
    u1_zero.vector()[:] = A0 * u1.vector()

    boundaryIntegral = assemble(u1_zero * ds)
    assert near(boundaryIntegral, 0.0)


@skip_in_parallel
def test_zero_columns_offdiag():
    """Test zero_columns applied to offdiagonal block"""
    mesh = UnitSquareMesh(MPI.comm_world, 20, 20)
    V = VectorFunctionSpace(mesh, "P", 2)
    Q = FunctionSpace(mesh, "P", 1)
    u = TrialFunction(V)
    q = TestFunction(Q)
    a = inner(div(u), q)*dx
    L = inner(Constant(0), q)*dx
    A = assemble(a)
    b = assemble(L)

    bc = DirichletBC(V, Constant((-32.23333, 43243.1)), 'on_boundary')

    # Compute residual with x satisfying bc before zero_columns
    u = Function(V)
    x = u.vector()
    bc.apply(x)
    r0 = A*x - b

    bc.zero_columns(A, b)

    # Test that A gets zero columns
    bc_dict = bc.get_boundary_values()
    for i in range(*A.local_range(0)):
        cols, vals = A.getrow(i)
        for j, v in zip(cols, vals):
            if j in bc_dict:
                assert v == 0.0

    # Compute residual with x satisfying bc after zero_columns
    # and check that it is preserved
    r1 = A*x - b
    assert numpy.isclose((r1-r0).norm('linf'), 0.0)


@skip_in_parallel
def test_zero_columns_square():
    """Test zero_columns applied to square matrix"""
    mesh = UnitSquareMesh(MPI.comm_world, 20, 20)
    V = FunctionSpace(mesh, "P", 1)
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v))*dx
    L = Constant(0)*v*dx
    A = assemble(a)
    b = assemble(L)
    u = Function(V)
    x = u.vector()

    bc = DirichletBC(V, 666.0, 'on_boundary')
    bc.zero_columns(A, b, 42.0)

    # Check that A gets zeros in bc rows and bc columns and 42 on
    # diagonal
    bc_dict = bc.get_boundary_values()
    for i in range(*A.local_range(0)):
        cols, vals = A.getrow(i)
        for j, v in zip(cols, vals):
            if i in bc_dict or j in bc_dict:
                if i == j:
                    assert numpy.isclose(v, 42.0)
                else:
                    assert v == 0.0

    # Check that solution of linear system works
    solve(A, x, b)
    assert numpy.isclose((b-A*x).norm('linf'), 0.0)
    x1 = x.copy()
    bc.apply(x1)
    x1 -= x
    assert numpy.isclose(x1.norm('linf'), 0.0)


def test_homogenize_consistency():
    mesh = UnitIntervalMesh(MPI.comm_world, 10)
    V = FunctionSpace(mesh, "CG", 1)

    for method in ['topological', 'geometric', 'pointwise']:
        bc = DirichletBC(V, Constant(0), "on_boundary", method=method)
        bc_new = DirichletBC(bc)
        bc_new.homogenize()
        assert bc_new.method() == bc.method()


def test_nocaching_values():
    """There might be caching of dof indices in DirichletBC.
    But caching of values is _not_ allowed."""
    mesh = UnitSquareMesh(MPI.comm_world, 4, 4)
    V = FunctionSpace(mesh, "P", 1)
    u = Function(V)
    x = u.vector()

    for method in ["topological", "geometric", "pointwise"]:
        bc = DirichletBC(V, 0.0, lambda x, b: True, method=method)

        x.set(0.0)
        bc.set_value(Constant(1.0))
        bc.apply(x)
        assert numpy.allclose(x.get_local(), 1.0)

        x.set(0.0)
        bc.set_value(Constant(2.0))
        bc.apply(x)
        assert numpy.allclose(x.get_local(), 2.0)
