#!/usr/bin/env py.test

"""Unit tests for the solve function"""

# Copyright (C) 2011 Anders Logg
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

import pytest
from dolfin import *
from dolfin_utils.test import *


@use_gc_barrier
def test_bcs():
    "Check that the bcs argument is picked up"

    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(100.0)

    a = dot(grad(u), grad(v))*dx + u*v*dx
    L = f*v*dx

    bc = DirichletBC(V, 0.0, DomainBoundary())

    # Single bc argument
    u1 = Function(V)
    solve(a == L, u1, bc)

    # List of bcs
    u2 = Function(V)
    solve(a == L, u2, [bc])

    # Single bc keyword argument
    u3 = Function(V)
    solve(a == L, u3, bcs=bc)

    # List of bcs keyword argument
    u4 = Function(V)
    solve(a == L, u4, bcs=[bc])

    # Check all solutions
    assert round(u1.vector().norm("l2") - 14.9362601686, 10) == 0
    assert round(u2.vector().norm("l2") - 14.9362601686, 10) == 0
    assert round(u3.vector().norm("l2") - 14.9362601686, 10) == 0
    assert round(u4.vector().norm("l2") - 14.9362601686, 10) == 0


@use_gc_barrier
def test_bcs_space():
    "Check that the bc space is checked to be a subspace of trial space"
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(100.0)

    a = dot(grad(u), grad(v))*dx + u*v*dx
    L = f*v*dx

    Q = FunctionSpace(mesh, "Lagrange", 2)
    bc = DirichletBC(Q, 0.0, DomainBoundary())

    u = Function(V)

    with pytest.raises(RuntimeError):
        solve(a == L, u, bc)

    with pytest.raises(RuntimeError):
        solve(action(a, u) - L == 0, u, bc)


@use_gc_barrier
def test_calling():
    "Test that unappropriate arguments are not allowed"
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(100.0)

    a = dot(grad(u), grad(v))*dx + u*v*dx
    L = f*v*dx

    bc = DirichletBC(V, 0.0, DomainBoundary())

    kwargs = {"solver_parameters":{"linear_solver": "lu"},
              "form_compiler_parameters":{"optimize": True}}

    A = assemble(a)
    b = assemble(L)
    x = Vector()

    with pytest.raises(RuntimeError):
        solve(A, x, b, **kwargs)

    # FIXME: Include more tests for this versatile function


@use_gc_barrier
def test_nonlinear_variational_solver_custom_comm():
    "Check that nonlinear variational solver works on subset of comm_world"
    if MPI.rank(mpi_comm_world()) == 0:
        mesh = UnitIntervalMesh(mpi_comm_self(), 2)
        V = FunctionSpace(mesh, "CG", 1)
        f = Constant(1)
        u = Function(V)
        v = TestFunction(V)
        F = inner(u, v)*dx - inner(f, v)*dx

        # Check that following does not deadlock
        solve(F == 0, u)
        solve(F == 0, u, solver_parameters={"nonlinear_solver": "newton"})
        if has_petsc():
            solve(F == 0, u, solver_parameters={"nonlinear_solver": "snes"})
