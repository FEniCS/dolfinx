# Copyright (C) 2018 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for Newton solver assembly"""

import numpy as np
from petsc4py import PETSc

import dolfin
import dolfin.fem as fem
import dolfin.function as function
import ufl
from ufl import derivative, dx, grad, inner


class NonlinearPDEProblem(dolfin.cpp.nls.NonlinearProblem):
    """Nonlinear problem class for a PDE problem."""

    def __init__(self, F, u, bc):
        super().__init__()
        V = u.function_space()
        du = function.TrialFunction(V)
        self.L = F
        self.a = derivative(F, u, du)
        self.bc = bc
        self._F, self._J = None, None

    def form(self, x):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def F(self, x):
        """Assemble residual vector."""
        if self._F is None:
            self._F = fem.assemble_vector(self.L)
        else:
            self._F = fem.assemble_vector(self._F, self.L)

        dolfin.fem.apply_lifting(self._F, [self.a], [[self.bc]], [x], -1.0)
        self._F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfin.fem.set_bc(self._F, [self.bc], x, -1.0)

        return self._F

    def J(self, x):
        """Assemble Jacobian matrix."""
        if self._J is None:
            self._J = fem.assemble_matrix(self.a, [self.bc])
        else:
            self._J = fem.assemble(self._J, self.a, [self.bc])
        return self._J


class NonlinearPDE_SNESProblem():
    def __init__(self, F, u, bc):
        super().__init__()
        V = u.function_space()
        du = function.TrialFunction(V)
        self.L = F
        self.a = derivative(F, u, du)
        self.a_comp = dolfin.fem.Form(self.a)
        self.bc = bc
        self._F, self._J = None, None
        self.u = u

    def F(self, snes, x, F):
        """Assemble residual vector."""
        _F = dolfin.cpp.la.PETScVector(F)
        _x = dolfin.cpp.la.PETScVector(x)
        _x.update_ghosts()
        x.copy(self.u.vector().vec())
        self.u.vector().update_ghosts()

        fem.assemble_vector(_F, self.L)
        fem.apply_lifting(_F, [self.a], [[self.bc]], [_x], -1.0)
        _F.vec().ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(_F, [self.bc], _x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        fem.assemble(J, self.a, [self.bc])


def test_linear_pde():
    """Test Newton solver for a linear PDE"""
    # Create mesh and function space
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 12, 12)
    V = dolfin.function.FunctionSpace(mesh, ("Lagrange", 1))
    u = dolfin.function.Function(V)
    v = function.TestFunction(V)
    F = inner(10.0, v) * dx - inner(grad(u), grad(v)) * dx

    def boundary(x):
        """Define Dirichlet boundary (x = 0 or x = 1)."""
        return np.logical_or(x[:, 0] < 1.0e-8, x[:, 0] > 1.0 - 1.0e-8)

    u_bc = function.Function(V)
    u_bc.vector().vec().set(1.0)
    u_bc.vector().update_ghosts()
    bc = fem.DirichletBC(V, u_bc, boundary)

    # Create nonlinear problem
    problem = NonlinearPDEProblem(F, u, bc)

    # Create Newton solver and solve
    solver = dolfin.cpp.nls.NewtonSolver(dolfin.MPI.comm_world)
    n, converged = solver.solve(problem, u.vector())
    assert converged
    assert n == 1

    # Increment boundary condition and solve again
    u_bc.vector().vec().set(2.0)
    u_bc.vector().update_ghosts()
    n, converged = solver.solve(problem, u.vector())
    assert converged
    assert n == 1


def test_nonlinear_pde():
    """Test Newton solver for a simple nonlinear PDE"""
    # Create mesh and function space
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 12, 5)
    V = dolfin.function.FunctionSpace(mesh, ("Lagrange", 1))
    u = dolfin.function.Function(V)
    v = function.TestFunction(V)
    F = inner(5.0, v) * dx - ufl.sqrt(u * u) * inner(
        grad(u), grad(v)) * dx - inner(u, v) * dx

    def boundary(x):
        """Define Dirichlet boundary (x = 0 or x = 1)."""
        return np.logical_or(x[:, 0] < 1.0e-8, x[:, 0] > 1.0 - 1.0e-8)

    u_bc = function.Function(V)
    u_bc.vector().vec().set(1.0)
    u_bc.vector().update_ghosts()
    bc = fem.DirichletBC(V, u_bc, boundary)

    # Create nonlinear problem
    problem = NonlinearPDEProblem(F, u, bc)

    # Create Newton solver and solve
    u.vector().vec().set(0.9)
    u.vector().update_ghosts()
    solver = dolfin.cpp.nls.NewtonSolver(dolfin.MPI.comm_world)
    n, converged = solver.solve(problem, u.vector())
    assert converged
    assert n < 6

    # Modify boundary condition and solve again
    u_bc.vector().vec().set(0.5)
    u_bc.vector().update_ghosts()
    n, converged = solver.solve(problem, u.vector())
    assert converged
    assert n < 6


def test_nonlinear_pde_snes():
    """Test Newton solver for a simple nonlinear PDE"""
    # Create mesh and function space
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 12, 15)
    V = dolfin.function.FunctionSpace(mesh, ("Lagrange", 1))
    u = dolfin.function.Function(V)
    v = function.TestFunction(V)
    F = inner(5.0, v) * dx - ufl.sqrt(u * u) * inner(
        grad(u), grad(v)) * dx - inner(u, v) * dx

    def boundary(x):
        """Define Dirichlet boundary (x = 0 or x = 1)."""
        return np.logical_or(x[:, 0] < 1.0e-8, x[:, 0] > 1.0 - 1.0e-8)

    u_bc = function.Function(V)
    u_bc.vector().vec().set(1.0)
    u_bc.vector().update_ghosts()
    bc = fem.DirichletBC(V, u_bc, boundary)

    # Create nonlinear problem
    problem = NonlinearPDE_SNESProblem(F, u, bc)

    u.vector().vec().set(0.9)
    u.vector().update_ghosts()

    b = dolfin.cpp.la.PETScVector(V.dofmap().index_map())
    J = dolfin.cpp.fem.create_matrix(problem.a_comp._cpp_object)

    # Create Newton solver and solve
    snes = PETSc.SNES().create()
    snes.setFunction(problem.F, b.vec())
    snes.setJacobian(problem.J, J)

    snes.setTolerances(rtol=1.0e-9, max_it=10)
    snes.setFromOptions()

    snes.getKSP().setTolerances(rtol=1.0e-9)
    snes.solve(None, u.vector().vec())
    assert snes.getConvergedReason() > 0
    assert snes.getIterationNumber() < 6

    # Modify boundary condition and solve again
    u_bc.vector().vec().set(0.5)
    u_bc.vector().update_ghosts()
    snes.solve(None, u.vector().vec())
    assert snes.getConvergedReason() > 0
    assert snes.getIterationNumber() < 6
    # print(snes.getIterationNumber())
    # print(snes.getFunctionNorm())


def test_newton_solver_inheritance():
    base = dolfin.cpp.nls.NewtonSolver(dolfin.MPI.comm_world)
    assert isinstance(base, dolfin.cpp.nls.NewtonSolver)

    class DerivedNewtonSolver(dolfin.cpp.nls.NewtonSolver):
        pass

    derived = DerivedNewtonSolver(dolfin.MPI.comm_world)
    assert isinstance(derived, DerivedNewtonSolver)
