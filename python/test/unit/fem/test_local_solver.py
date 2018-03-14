"""Unit tests for LocalSolver"""

# Copyright (C) 2013 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
import numpy
from dolfin import *
from dolfin_utils.test import skip_in_parallel
from dolfin_utils.test import set_parameters_fixture
ghost_mode = set_parameters_fixture("ghost_mode", ["shared_facet"])


def test_solve_global_rhs():
    mesh = UnitCubeMesh(2, 3, 3)
    V = FunctionSpace(mesh, "Discontinuous Lagrange", 2)
    W = FunctionSpace(mesh, "Lagrange", 2)

    u, v = TrialFunction(V), TestFunction(V)
    f = Expression("x[0]*x[0] + x[0]*x[1] + x[1]*x[1]", element=W.ufl_element())

    # Forms for projection
    a, L = inner(v, u)*dx, inner(v, f)*dx

    solvers = [LocalSolver.SolverType.LU, LocalSolver.SolverType.Cholesky]
    for solver_type in solvers:

        # First solve
        u = Function(V)
        local_solver = LocalSolver(a, L, solver_type)
        local_solver.solve_global_rhs(u)
        error = assemble((u - f)*(u - f)*dx)
        assert round(error, 10) == 0

        # Test cached factorization
        u.vector().zero()
        local_solver.factorize()
        local_solver.solve_global_rhs(u)
        error = assemble((u - f)*(u - f)*dx)
        assert round(error, 10) == 0

        # Clear cache and re-compute
        u.vector().zero()
        local_solver.clear_factorization()
        local_solver.solve_global_rhs(u)
        error = assemble((u - f)*(u - f)*dx)
        assert round(error, 10) == 0


def test_solve_local_rhs(ghost_mode):
    mesh = UnitCubeMesh(1, 5, 1)
    V = FunctionSpace(mesh, "Lagrange", 2)
    W = FunctionSpace(mesh, "Lagrange", 2)

    u, v = TrialFunction(V), TestFunction(V)
    f = Constant(10.0)

    # Forms for projection
    a, L = inner(v, u)*dx, inner(v, f)*dx

    solvers = [LocalSolver.SolverType.LU, LocalSolver.SolverType.Cholesky]
    for solver_type in solvers:

        # First solve
        u = Function(V)
        local_solver = LocalSolver(a, L, solver_type)
        local_solver.solve_local_rhs(u)
        x = u.vector().copy()
        x[:] = 10.0
        assert round((u.vector() - x).norm("l2") - 0.0, 10) == 0

        u.vector().zero()
        local_solver.factorize()
        local_solver.solve_local_rhs(u)
        assert round((u.vector() - x).norm("l2") - 0.0, 10) == 0

        u.vector().zero()
        local_solver.clear_factorization()
        local_solver.solve_local_rhs(u)
        assert round((u.vector() - x).norm("l2") - 0.0, 10) == 0


def test_solve_local_rhs_facet_integrals(ghost_mode):
    mesh = UnitSquareMesh(4, 4)

    # Facet function is used here to verify that the proper domains
    # of the rhs are used unlike before where the rhs domains were
    # taken to be the same as the lhs domains
    marker = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
    ds0 = Measure("ds", domain=mesh, subdomain_data=marker, subdomain_id=0)

    Vu = VectorFunctionSpace(mesh, 'DG', 1)
    Vv = FunctionSpace(mesh, 'DGT', 1)
    u = TrialFunction(Vu)
    v = TestFunction(Vv)

    n = FacetNormal(mesh)
    w = Constant([1, 1])

    a = dot(u, n)*v*ds
    L = dot(w, n)*v*ds0

    for R in '+-':
        a += dot(u(R), n(R))*v(R)*dS
        L += dot(w(R), n(R))*v(R)*dS

    u = Function(Vu)
    local_solver = LocalSolver(a, L)
    local_solver.solve_local_rhs(u)

    x = u.vector().copy()
    x[:] = 1
    assert round((u.vector() - x).norm('l2'), 10) == 0


def test_local_solver_dg(ghost_mode):
    mesh = UnitIntervalMesh(50)
    U = FunctionSpace(mesh, "DG", 2)

    # Set initial values
    u0 = interpolate(Expression("cos(pi*x[0])", degree=2), U)

    # Define test and trial functions
    v, u = TestFunction(U), TrialFunction(U)

    # Set time step size
    dt = Constant(2.0e-4)

    # Define fluxes on interior and exterior facets
    u_hat = avg(u0) + 0.25*jump(u0)
    u_hatbnd = -u0 + 0.25*(u0 - 1.0)

    # Define variational formulation
    a = u*v*dx
    L = (u0*v + dt*u0*v.dx(0))*dx - dt*u_hat*jump(v)*dS - dt*u_hatbnd*v*ds

    # Compute reference solution with global LU solver
    u_lu = Function(U)
    solve(a == L, u_lu, solver_parameters = {"linear_solver" : "lu"})

    # Compute solution with local solver and compare
    local_solver = LocalSolver(a, L)
    u_ls = Function(U)
    local_solver.solve_global_rhs(u_ls)
    assert round((u_lu.vector() - u_ls.vector()).norm("l2"), 12) == 0

    # Compute solution with local solver (Cholesky) and compare
    local_solver = LocalSolver(a, L, LocalSolver.SolverType.Cholesky)
    u_ls = Function(U)
    local_solver.solve_global_rhs(u_ls)
    assert round((u_lu.vector() - u_ls.vector()).norm("l2"), 12) == 0


def test_solve_local(ghost_mode):
    mesh = UnitIntervalMesh(50)
    U = FunctionSpace(mesh, "DG", 2)

    # Set initial values
    u0 = interpolate(Expression("cos(pi*x[0])", degree=2), U)

    # Define test and trial functions
    v, u = TestFunction(U), TrialFunction(U)

    # Set time step size
    dt = Constant(2.0e-4)

    # Define fluxes on interior and exterior facets
    u_hat = avg(u0) + 0.25*jump(u0)
    u_hatbnd = -u0 + 0.25*(u0 - 1.0)

    # Define variational formulation
    a = u*v*dx
    L = (u0*v + dt*u0*v.dx(0))*dx - dt*u_hat*jump(v)*dS - dt*u_hatbnd*v*ds
    b = assemble(L)

    # Compute reference solution with global LU solver
    u_lu = Function(U)
    solve(a == L, u_lu, solver_parameters = {"linear_solver" : "lu"})

    # Compute solution with local solver and compare
    local_solver = LocalSolver(a)
    u_ls = Function(U)
    local_solver.solve_local(u_ls.vector(), b, U.dofmap())
    assert round((u_lu.vector() - u_ls.vector()).norm("l2"), 12) == 0

    # Compute solution with local solver (Cholesky) and compare
    local_solver = LocalSolver(a, solver_type=LocalSolver.SolverType.Cholesky)
    u_ls = Function(U)
    local_solver.solve_local(u_ls.vector(), b, U.dofmap())
    assert round((u_lu.vector() - u_ls.vector()).norm("l2"), 12) == 0
