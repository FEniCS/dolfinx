# Copyright (C) 2018 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for Newton solver assembly"""

import numpy as np

import ufl
from dolfinx import cpp as _cpp
from dolfinx import la
from dolfinx.fem import (Function, FunctionSpace, dirichletbc, form,
                         locate_dofs_geometrical)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_matrix, create_vector, set_bc)
from dolfinx.mesh import create_unit_square
from ufl import TestFunction, TrialFunction, derivative, dx, grad, inner

from mpi4py import MPI
from petsc4py import PETSc


class NonlinearPDEProblem:
    """Nonlinear problem class for a PDE problem."""

    def __init__(self, F, u, bc):
        V = u.function_space
        du = TrialFunction(V)
        self.L = form(F)
        self.a = form(derivative(F, u, du))
        self.bc = bc

    def form(self, x):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def F(self, x, b):
        """Assemble residual vector."""
        with b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(b, self.L)
        apply_lifting(b, [self.a], bcs=[[self.bc]], x0=[x], scale=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [self.bc], x, -1.0)

    def J(self, x, A):
        """Assemble Jacobian matrix."""
        A.zeroEntries()
        assemble_matrix(A, self.a, bcs=[self.bc])
        A.assemble()

    def matrix(self):
        return create_matrix(self.a)

    def vector(self):
        return create_vector(self.L)


class NonlinearPDE_SNESProblem:
    def __init__(self, F, u, bc):
        V = u.function_space
        du = TrialFunction(V)
        self.L = form(F)
        self.a = form(derivative(F, u, du))
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
        assemble_vector(F, self.L)
        apply_lifting(F, [self.a], bcs=[[self.bc]], x0=[x], scale=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, [self.bc], x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        J.zeroEntries()
        assemble_matrix(J, self.a, bcs=[self.bc])
        J.assemble()


def test_linear_pde():
    """Test Newton solver for a linear PDE"""
    # Create mesh and function space
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u = Function(V)
    v = TestFunction(V)
    F = inner(10.0, v) * dx - inner(grad(u), grad(v)) * dx

    bc = dirichletbc(PETSc.ScalarType(1.0),
                     locate_dofs_geometrical(V, lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                        np.isclose(x[0], 1.0))), V)

    # Create nonlinear problem
    problem = NonlinearPDEProblem(F, u, bc)

    # Create Newton solver and solve
    solver = _cpp.nls.petsc.NewtonSolver(MPI.COMM_WORLD)
    solver.setF(problem.F, problem.vector())
    solver.setJ(problem.J, problem.matrix())
    solver.set_form(problem.form)
    n, converged = solver.solve(u.vector)
    assert converged
    assert n == 1

    # Increment boundary condition and solve again
    bc.g.value[...] = PETSc.ScalarType(2.0)
    n, converged = solver.solve(u.vector)
    assert converged
    assert n == 1


def test_nonlinear_pde():
    """Test Newton solver for a simple nonlinear PDE"""
    # Create mesh and function space
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 5)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u = Function(V)
    v = TestFunction(V)
    F = inner(5.0, v) * dx - ufl.sqrt(u * u) * inner(
        grad(u), grad(v)) * dx - inner(u, v) * dx

    bc = dirichletbc(PETSc.ScalarType(1.0),
                     locate_dofs_geometrical(V, lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                        np.isclose(x[0], 1.0))), V)

    # Create nonlinear problem
    problem = NonlinearPDEProblem(F, u, bc)

    # Create Newton solver and solve
    u.x.array[:] = 0.9
    solver = _cpp.nls.petsc.NewtonSolver(MPI.COMM_WORLD)
    solver.setF(problem.F, problem.vector())
    solver.setJ(problem.J, problem.matrix())
    solver.set_form(problem.form)
    n, converged = solver.solve(u.vector)
    assert converged
    assert n < 6

    # Modify boundary condition and solve again
    bc.g.value[...] = 0.5
    n, converged = solver.solve(u.vector)
    assert converged
    assert n > 0 and n < 6


def test_nonlinear_pde_snes():
    """Test Newton solver for a simple nonlinear PDE"""
    # Create mesh and function space
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 15)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u = Function(V)
    v = TestFunction(V)
    F = inner(5.0, v) * dx - ufl.sqrt(u * u) * inner(
        grad(u), grad(v)) * dx - inner(u, v) * dx

    u_bc = Function(V)
    u_bc.x.array[:] = 1.0
    bc = dirichletbc(u_bc, locate_dofs_geometrical(V, lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                              np.isclose(x[0], 1.0))))

    # Create nonlinear problem
    problem = NonlinearPDE_SNESProblem(F, u, bc)

    u.x.array[:] = 0.9
    b = la.create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
    J = create_matrix(problem.a)

    # Create Newton solver and solve
    snes = PETSc.SNES().create()
    snes.setFunction(problem.F, b)
    snes.setJacobian(problem.J, J)

    snes.setTolerances(rtol=1.0e-9, max_it=10)
    snes.getKSP().setType("preonly")
    snes.getKSP().setTolerances(rtol=1.0e-9)
    snes.getKSP().getPC().setType("lu")

    snes.solve(None, u.vector)
    assert snes.getConvergedReason() > 0
    assert snes.getIterationNumber() < 6

    # Modify boundary condition and solve again
    u_bc.x.array[:] = 0.6
    snes.solve(None, u.vector)
    assert snes.getConvergedReason() > 0
    assert snes.getIterationNumber() < 6
    # print(snes.getIterationNumber())
    # print(snes.getFunctionNorm())
