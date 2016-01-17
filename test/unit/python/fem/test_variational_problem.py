#!/usr/bin/env py.test

"""Unit tests for the Nonlinear- and Linear-VariationalProblem classes"""

# Copyright (C) 2016 Garth N. Wells
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
def test_linear_construction():
    "Test construction of LinearVariationalProblem"

    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(100.0)

    a = dot(grad(u), grad(v))*dx + u*v*dx
    L = f*v*dx

    bc = DirichletBC(V, 0.0, DomainBoundary())

    w = Function(V)
    with pytest.raises(TypeError):
        problem = LinearVariationalProblem(a, L)
    with pytest.raises(RuntimeError):
        problem = LinearVariationalProblem(a, L, [bc])
    with pytest.raises(RuntimeError):
        problem = LinearVariationalProblem(a, L, [bc], w)
    problem = LinearVariationalProblem(a, L, w, [])
    problem = LinearVariationalProblem(a, L, w, [bc])
    problem = LinearVariationalProblem(a, L, w, [bc, bc])


@use_gc_barrier
def test_nonlinear_construction():
    "Test construction of NonlinearVariationalProblem"

    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = Function(V)
    du = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(100.0)

    F = dot(grad(u), grad(v))*dx + u*v*dx - f*v*dx
    J = derivative(F, u, du)
    bc = DirichletBC(V, 0.0, DomainBoundary())

    with pytest.raises(RuntimeError):
        problem = NonlinearVariationalProblem(F, u, J)
    problem = NonlinearVariationalProblem(F, u)
    problem = NonlinearVariationalProblem(F, u, [])
    problem = NonlinearVariationalProblem(F, u, [], J)
    problem = NonlinearVariationalProblem(F, u, J=J, bcs=[])
