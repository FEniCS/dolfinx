# Copyright (C) 2018 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for Newton solver assembly"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem as fem
import ufl
from dolfinx import function
from ufl import TestFunction, TrialFunction, derivative, dx, grad, inner


class NonlinearPDEProblem(dolfinx.cpp.nls.NonlinearProblem):
    """Nonlinear problem class for a PDE problem."""

    def __init__(self, F, u, bc):
        super().__init__()
        V = u.function_space
        du = TrialFunction(V)
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
            with self._F.localForm() as f_local:
                f_local.set(0.0)
            self._F = fem.assemble_vector(self._F, self.L)
        dolfinx.fem.apply_lifting(self._F, [self.a], [[self.bc]], [x], -1.0)
        self._F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(self._F, [self.bc], x, -1.0)

        return self._F

    def J(self, x):
        """Assemble Jacobian matrix."""
        if self._J is None:
            self._J = fem.assemble_matrix(self.a, [self.bc])
        else:
            self._J.zeroEntries()
            self._J = fem.assemble_matrix(self._J, self.a, [self.bc])
        self._J.assemble()
        return self._J


class NonlinearPDE_SNESProblem():
    def __init__(self, F, u, bc):
        super().__init__()
        V = u.function_space
        du = TrialFunction(V)
        self.L = F
        self.a = derivative(F, u, du)
        self.a_comp = dolfinx.fem.Form(self.a)
        self.bc = bc
        self._F, self._J = None, None
        self.u = u

    def F(self, snes, x, F):
        """Assemble residual vector."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        with F.localForm() as f_local:
            f_local.set(0.0)
        fem.assemble_vector(F, self.L)
        fem.apply_lifting(F, [self.a], [[self.bc]], [x], -1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(F, [self.bc], x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        J.zeroEntries()
        fem.assemble_matrix(J, self.a, [self.bc])
        J.assemble()


def test_linear_pde():
    """Test Newton solver for a linear PDE"""
    # Create mesh and function space
    mesh = dolfinx.generation.UnitSquareMesh(MPI.COMM_WORLD, 12, 12)
    V = function.FunctionSpace(mesh, ("Lagrange", 1))
    u = function.Function(V)
    v = TestFunction(V)
    F = inner(10.0, v) * dx - inner(grad(u), grad(v)) * dx

    def boundary(x):
        """Define Dirichlet boundary (x = 0 or x = 1)."""
        return np.logical_or(x[0] < 1.0e-8, x[0] > 1.0 - 1.0e-8)

    u_bc = function.Function(V)
    u_bc.vector.set(1.0)
    u_bc.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    bc = fem.DirichletBC(u_bc, fem.locate_dofs_geometrical(V, boundary))

    # Create nonlinear problem
    problem = NonlinearPDEProblem(F, u, bc)

    # Create Newton solver and solve
    solver = dolfinx.cpp.nls.NewtonSolver(MPI.COMM_WORLD)
    n, converged = solver.solve(problem, u.vector)
    assert converged
    assert n == 1

    # Increment boundary condition and solve again
    u_bc.vector.set(2.0)
    u_bc.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    n, converged = solver.solve(problem, u.vector)
    assert converged
    assert n == 1


def test_nonlinear_pde():
    """Test Newton solver for a simple nonlinear PDE"""
    # Create mesh and function space
    mesh = dolfinx.generation.UnitSquareMesh(MPI.COMM_WORLD, 12, 5)
    V = function.FunctionSpace(mesh, ("Lagrange", 1))
    u = dolfinx.function.Function(V)
    v = TestFunction(V)
    F = inner(5.0, v) * dx - ufl.sqrt(u * u) * inner(
        grad(u), grad(v)) * dx - inner(u, v) * dx

    def boundary(x):
        """Define Dirichlet boundary (x = 0 or x = 1)."""
        return np.logical_or(x[0] < 1.0e-8, x[0] > 1.0 - 1.0e-8)

    u_bc = function.Function(V)
    u_bc.vector.set(1.0)
    u_bc.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    bc = fem.DirichletBC(u_bc, fem.locate_dofs_geometrical(V, boundary))

    # Create nonlinear problem
    problem = NonlinearPDEProblem(F, u, bc)

    # Create Newton solver and solve
    u.vector.set(0.9)
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    solver = dolfinx.cpp.nls.NewtonSolver(MPI.COMM_WORLD)
    n, converged = solver.solve(problem, u.vector)
    assert converged
    assert n < 6

    # Modify boundary condition and solve again
    u_bc.vector.set(0.5)
    u_bc.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    n, converged = solver.solve(problem, u.vector)
    assert converged
    assert n < 6


def test_nonlinear_pde_snes():
    """Test Newton solver for a simple nonlinear PDE"""
    # Create mesh and function space
    mesh = dolfinx.generation.UnitSquareMesh(MPI.COMM_WORLD, 12, 15)
    V = function.FunctionSpace(mesh, ("Lagrange", 1))
    u = function.Function(V)
    v = TestFunction(V)
    F = inner(5.0, v) * dx - ufl.sqrt(u * u) * inner(
        grad(u), grad(v)) * dx - inner(u, v) * dx

    def boundary(x):
        """Define Dirichlet boundary (x = 0 or x = 1)."""
        return np.logical_or(x[0] < 1.0e-8, x[0] > 1.0 - 1.0e-8)

    u_bc = function.Function(V)
    u_bc.vector.set(1.0)
    u_bc.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    bc = fem.DirichletBC(u_bc, fem.locate_dofs_geometrical(V, boundary))

    # Create nonlinear problem
    problem = NonlinearPDE_SNESProblem(F, u, bc)

    u.vector.set(0.9)
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    b = dolfinx.cpp.la.create_vector(V.dofmap.index_map)
    J = dolfinx.cpp.fem.create_matrix(problem.a_comp._cpp_object)

    # Create Newton solver and solve
    snes = PETSc.SNES().create()
    snes.setFunction(problem.F, b)
    snes.setJacobian(problem.J, J)

    snes.setTolerances(rtol=1.0e-9, max_it=10)
    snes.getKSP().setType("preonly")
    snes.getKSP().setTolerances(rtol=1.0e-9)

    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType("superlu_dist")

    snes.solve(None, u.vector)
    assert snes.getConvergedReason() > 0
    assert snes.getIterationNumber() < 6

    # Modify boundary condition and solve again
    u_bc.vector.set(0.5)
    u_bc.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    snes.solve(None, u.vector)
    assert snes.getConvergedReason() > 0
    assert snes.getIterationNumber() < 6
    # print(snes.getIterationNumber())
    # print(snes.getFunctionNorm())


def test_newton_solver_inheritance():
    base = dolfinx.cpp.nls.NewtonSolver(MPI.COMM_WORLD)
    assert isinstance(base, dolfinx.cpp.nls.NewtonSolver)

    class DerivedNewtonSolver(dolfinx.cpp.nls.NewtonSolver):
        pass

    derived = DerivedNewtonSolver(MPI.COMM_WORLD)
    assert isinstance(derived, DerivedNewtonSolver)


def test_newton_solver_inheritance_override_methods():
    import functools
    called_methods = {}

    def check_is_called(method):
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            called_methods[method.__name__] = True
            return method(*args, **kwargs)
        return wrapper

    class CustomNewtonSolver(dolfinx.cpp.nls.NewtonSolver):

        def __init__(self, comm):
            super().__init__(comm)

        @check_is_called
        def update_solution(self, x, dx, relaxation,
                            problem, it):
            return super().update_solution(x, dx, relaxation, problem, it)

        @check_is_called
        def converged(self, r, problem, it):
            return super().converged(r, problem, it)

    mesh = dolfinx.generation.UnitSquareMesh(MPI.COMM_WORLD, 12, 12)
    V = function.FunctionSpace(mesh, ("Lagrange", 1))
    u = function.Function(V)
    v = TestFunction(V)
    F = inner(10.0, v) * dx - inner(grad(u), grad(v)) * dx

    def boundary(x):
        """Define Dirichlet boundary (x = 0 or x = 1)."""
        return np.logical_or(x[0] < 1.0e-8, x[0] > 1.0 - 1.0e-8)

    u_bc = function.Function(V)
    bc = fem.DirichletBC(u_bc, fem.locate_dofs_geometrical(V, boundary))

    # Create nonlinear problem
    problem = NonlinearPDEProblem(F, u, bc)

    # Create Newton solver and solve
    solver = CustomNewtonSolver(MPI.COMM_WORLD)
    n, converged = solver.solve(problem, u.vector)

    assert called_methods[CustomNewtonSolver.converged.__name__]
    assert called_methods[CustomNewtonSolver.update_solution.__name__]
