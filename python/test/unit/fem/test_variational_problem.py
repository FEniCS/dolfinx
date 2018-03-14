"""Unit tests for the Nonlinear- and Linear-VariationalProblem classes"""

# Copyright (C) 2016 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
from dolfin import *
from dolfin_utils.test import *


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
    with pytest.raises(Exception):
        problem = LinearVariationalProblem(a, L, [bc])
    with pytest.raises(Exception):
        problem = LinearVariationalProblem(a, L, [bc], w)
    problem = LinearVariationalProblem(a, L, w, [])
    problem = LinearVariationalProblem(a, L, w, [bc])
    problem = LinearVariationalProblem(a, L, w, [bc, bc])


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

    with pytest.raises(Exception):
        problem = NonlinearVariationalProblem(F, u, J)
    problem = NonlinearVariationalProblem(F, u)
    problem = NonlinearVariationalProblem(F, u, [])
    problem = NonlinearVariationalProblem(F, u, [], J)
    problem = NonlinearVariationalProblem(F, u, J=J, bcs=[])
