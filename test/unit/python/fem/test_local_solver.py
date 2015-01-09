#!/usr/bin/env py.test

"""Unit tests for LocalSolver"""

# Copyright (C) 2013 Garth N. Wells
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
#
# Modified by Steven Vandekerckhove, 2014.
#
# First added:  2013-02-13
# Last changed: 2014-11-18

import pytest
import numpy
from dolfin import *

def test_local_solver():
    mesh = UnitCubeMesh(2, 3, 3)
    V = FunctionSpace(mesh, "Discontinuous Lagrange", 2)
    W = FunctionSpace(mesh, "Lagrange", 2)

    u, v = TrialFunction(V), TestFunction(V)
    f = Expression("x[0]*x[0] + x[0]*x[1] + x[1]*x[1]", element=W.ufl_element())

    # Forms for projection
    a, L = inner(v, u)*dx, inner(v, f)*dx

    # Wrap forms as DOLFIN forms (LocalSolver hasn't been properly
    # wrapped in Python yet)
    a, L = Form(a), Form(L)

    u = Function(V)
    local_solver = cpp.LocalSolver(a, L)
    local_solver.solve(u.vector())
    error = assemble((u - f)*(u - f)*dx)
    assert round(error, 10) == 0

    u = Function(V)
    local_solver = cpp.LocalSolver(a, L)
    local_solver.factorize()
    local_solver.solve(u.vector())
    error = assemble((u - f)*(u - f)*dx)
    assert round(error, 10) == 0


def xtest_local_solver_reuse_factorization():

    mesh = UnitCubeMesh(16, 16, 16)
    V = FunctionSpace(mesh, "DG", 2)

    v = TestFunction(V)
    u = TrialFunction(V)
    f = Constant(10.0)

    # Forms for projection
    a = inner(v, u)*dx
    L = inner(v, f)*dx

    u = Function(V)
    local_solver = LocalSolver(a, L)
    local_solver.solve(u.vector())
    x = u.vector().copy()
    x[:] = 10.0
    assert round((u.vector() - x).norm("l2") - 0.0, 9) == 0

def xtest_local_solver_dg():
    # Prepare a mesh
    mesh = UnitIntervalMesh(50)

    # Define function space
    U = FunctionSpace(mesh, "DG", 2)

    # Set some expressions
    uinit = Expression("cos(pi*x[0])")
    ubdr  = Constant("1.0")

    # Set initial values
    u0 = interpolate(uinit, U)

    # Define test and trial functions
    v = TestFunction(U)
    u = TrialFunction(U)

    # Set time step size
    DT = Constant(2.e-4)

    # Define fluxes on interior and exterior facets
    uhat    = avg(u0) + 0.25*jump(u0)
    uhatbnd = -u0 + .25*(u0-ubdr)

    # Define variational formulation
    a = u*v*dx
    L = (u0*v + DT*u0*v.dx(0))*dx \
        -DT* uhat * jump(v)*dS \
        -DT* uhatbnd * v*ds

    # Prepare solution
    u_lu = Function(U)
    u_ls = Function(U)

    # Compute reference with global LU solver
    solve(a == L, u_lu, solver_parameters = {"linear_solver" : "lu"})

    # Prepare LocalSolver
    local_solver = LocalSolver(a, L)
    local_solver.solve(u_ls.vector())

    assert (u_lu.vector() - u_ls.vector()).norm("l2") < 1e-14

def xtest_local_solver_dg_solve_xb():
    # Prepare a mesh
    mesh = UnitIntervalMesh(50)

    # Define function space
    U = FunctionSpace(mesh, "DG", 2)

    # Set some expressions
    uinit = Expression("cos(pi*x[0])")
    ubdr  = Constant("1.0")

    # Set initial values
    u0 = interpolate(uinit, U)

    # Define test and trial functions
    v = TestFunction(U)
    u = TrialFunction(U)

    # Set time step size
    DT = Constant(2.e-4)

    # Define fluxes on interior and exterior facets
    uhat    = avg(u0) + 0.25*jump(u0)
    uhatbnd = -u0 + .25*(u0-ubdr)

    # Define variational formulation
    a = u*v*dx
    L = (u0*v + DT*u0*v.dx(0))*dx \
        -DT* uhat * jump(v)*dS \
        -DT* uhatbnd * v*ds

    # Prepare solution
    u_lu = Function(U)
    u_ls = Function(U)

    # Compute reference with global LU solver
    solve(a == L, u_lu, solver_parameters = {"linear_solver" : "lu"})

    # Prepare LocalSolver
    local_solver = LocalSolver(a)
    b = assemble(L)
    local_solver.solve(u_ls.vector(), b)

    assert (u_lu.vector() - u_ls.vector()).norm("l2") < 1e-14
