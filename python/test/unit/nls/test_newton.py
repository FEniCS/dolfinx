# Copyright (C) 2018 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for Newton solver assembly"""

import numpy as np

import dolfin
import dolfin.fem as fem
import dolfin.function as function
from ufl import derivative, dx, grad, inner


def test_linear_pde():
    class PoissonEquation(dolfin.cpp.nls.NonlinearProblem):
        """Nonlinear problem class for the linear Poisson problem."""

        def __init__(self, u):
            super().__init__()
            V = u.function_space()
            du = function.TrialFunction(V)
            v = function.TestFunction(V)
            self.L = inner(10.0, v) * dx - inner(grad(u), grad(v)) * dx
            self.a = derivative(self.L, u, du)

            def boundary(x):
                """Define Dirichlet boundary (x = 0 or x = 1)."""
                return np.logical_or(x[:, 0] < 1.0e-8, x[:, 0] > 1.0 - 1.0e-8)

            u_bc = function.Function(V)
            u_bc.vector().set(1.0)
            self.bc = fem.DirichletBC(V, u_bc, boundary)

            self._F, self._J = None, None

        def F(self, x):
            """Assemble residual vector."""
            if self._F is None:
                self._F = fem.assemble_vector(
                    [self.L], [[self.a]], [self.bc],
                    dolfin.cpp.fem.BlockType.monolithic, x)
            else:
                self._F = fem.assemble(self._F, self.L, [self.a], [self.bc], x)
            return self._F

        def J(self, x):
            """Assemble Jacobian matrix."""
            if self._J is None:
                self._J = fem.assemble_matrix(
                    [[self.a]], [self.bc], dolfin.cpp.fem.BlockType.monolithic)
            else:
                self._J = fem.assemble(self._J, self.a, [self.bc])
            return self._J

    # Create mesh and function space
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 12, 12)
    V = dolfin.function.FunctionSpace(mesh, ("Lagrange", 1))

    # Create solution function and nonlinear problem
    u = dolfin.function.Function(V)
    problem = PoissonEquation(u)

    # Create Newton solver and solve
    solver = dolfin.cpp.nls.NewtonSolver(dolfin.MPI.comm_world)
    n, converged = solver.solve(problem, u.vector())
    assert converged is True
    assert n == 1


def test_nonlinear_pde():
    class PoissonEquation(dolfin.cpp.nls.NonlinearProblem):
        """Nonlinear problem class for the linear Poisson problem."""

        def __init__(self, u):
            super().__init__()
            V = u.function_space()
            du = function.TrialFunction(V)
            v = function.TestFunction(V)
            self.L = inner(10.0, v) * dx - u*u*inner(grad(u), grad(v)) * dx
            self.a = derivative(self.L, u, du)

            def boundary(x):
                """Define Dirichlet boundary (x = 0 or x = 1)."""
                return np.logical_or(x[:, 0] < 1.0e-8, x[:, 0] > 1.0 - 1.0e-8)

            u_bc = function.Function(V)
            u_bc.vector().set(0.0)
            self.bc = fem.DirichletBC(V, u_bc, boundary)

            self._F, self._J = None, None

        def F(self, x):
            """Assemble residual vector."""
            if self._F is None:
                self._F = fem.assemble_vector(
                    [self.L], [[self.a]], [self.bc],
                    dolfin.cpp.fem.BlockType.monolithic, x)
            else:
                x.vec().view()
                self._F = fem.assemble(self._F, self.L, [self.a], [self.bc], x)
                self._F.vec().view()
            return self._F

        def J(self, x):
            """Assemble Jacobian matrix."""
            if self._J is None:
                self._J = fem.assemble_matrix(
                    [[self.a]], [self.bc], dolfin.cpp.fem.BlockType.monolithic)
            else:
                self._J = fem.assemble(self._J, self.a, [self.bc])
            return self._J

    # Create mesh and function space
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 2, 1)
    V = dolfin.function.FunctionSpace(mesh, ("Lagrange", 1))

    # Create solution function and nonlinear problem
    u = dolfin.function.Function(V)
    problem = PoissonEquation(u)

    # Create Newton solver and solve
    solver = dolfin.cpp.nls.NewtonSolver(dolfin.MPI.comm_world)
    n, converged = solver.solve(problem, u.vector())
    print(n, converged)
    # assert converged is True
    # assert n == 1
