# Copyright (C) 2018 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for Newton solver assembly"""

import dolfinx
import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc
from ufl import TestFunction, TrialFunction, derivative, dx, grad, inner


class NonlinearPDEProblem:
    """Nonlinear problem class for a PDE problem."""

    def __init__(self, F, u, bc):
        V = u.function_space
        du = TrialFunction(V)
        self.L = F
        self.a = derivative(F, u, du)
        self.bc = bc

    def form(self, x):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def F(self, x, b):
        """Assemble residual vector."""
        with b.localForm() as b_local:
            b_local.set(0.0)
        fem.assemble_vector(b, self.L)
        dolfinx.fem.apply_lifting(b, [self.a], bcs=[[self.bc]], x0=[x], scale=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(b, [self.bc], x, -1.0)

    def J(self, x, A):
        """Assemble Jacobian matrix."""
        A.zeroEntries()
        fem.assemble_matrix(A, self.a, bcs=[self.bc])
        A.assemble()

    def matrix(self):
        return fem.create_matrix(self.a)

    def vector(self):
        return fem.create_vector(self.L)


class NonlinearPDE_SNESProblem:
    def __init__(self, F, u, bc):
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
        fem.apply_lifting(F, [self.a], bcs=[[self.bc]], x0=[x], scale=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(F, [self.bc], x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        J.zeroEntries()
        fem.assemble_matrix(J, self.a, bcs=[self.bc])
        J.assemble()


def test_linear_pde():
    """Test Newton solver for a linear PDE"""
    # Create mesh and function space
    mesh = dolfinx.generation.UnitSquareMesh(MPI.COMM_WORLD, 12, 12)
    V = fem.FunctionSpace(mesh, ("Lagrange", 1))
    u = fem.Function(V)
    v = TestFunction(V)
    F = inner(10.0, v) * dx - inner(grad(u), grad(v)) * dx

    def boundary(x):
        """Define Dirichlet boundary (x = 0 or x = 1)."""
        return np.logical_or(x[0] < 1.0e-8, x[0] > 1.0 - 1.0e-8)

    u_bc = fem.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(1.0)
    bc = fem.DirichletBC(u_bc, fem.locate_dofs_geometrical(V, boundary))

    # Create nonlinear problem
    problem = NonlinearPDEProblem(F, u, bc)

    # Create Newton solver and solve
    solver = dolfinx.cpp.nls.NewtonSolver(MPI.COMM_WORLD)
    solver.setF(problem.F, problem.vector())
    solver.setJ(problem.J, problem.matrix())
    solver.set_form(problem.form)
    n, converged = solver.solve(u.vector)
    assert converged
    assert n == 1

    # Increment boundary condition and solve again
    with u_bc.vector.localForm() as u_local:
        u_local.set(2.0)
    n, converged = solver.solve(u.vector)
    assert converged
    assert n == 1


def test_nonlinear_pde():
    """Test Newton solver for a simple nonlinear PDE"""
    # Create mesh and function space
    mesh = dolfinx.generation.UnitSquareMesh(MPI.COMM_WORLD, 12, 5)
    V = fem.FunctionSpace(mesh, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    v = TestFunction(V)
    F = inner(5.0, v) * dx - ufl.sqrt(u * u) * inner(
        grad(u), grad(v)) * dx - inner(u, v) * dx

    def boundary(x):
        """Define Dirichlet boundary (x = 0 or x = 1)."""
        return np.logical_or(x[0] < 1.0e-8, x[0] > 1.0 - 1.0e-8)

    u_bc = fem.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(1.0)
    bc = fem.DirichletBC(u_bc, fem.locate_dofs_geometrical(V, boundary))

    # Create nonlinear problem
    problem = NonlinearPDEProblem(F, u, bc)

    # Create Newton solver and solve
    with u.vector.localForm() as u_local:
        u_local.set(0.9)
    solver = dolfinx.cpp.nls.NewtonSolver(MPI.COMM_WORLD)
    solver.setF(problem.F, problem.vector())
    solver.setJ(problem.J, problem.matrix())
    solver.set_form(problem.form)
    n, converged = solver.solve(u.vector)
    assert converged
    assert n < 6

    # Modify boundary condition and solve again
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.5)
    n, converged = solver.solve(u.vector)
    assert converged
    assert n < 6


def test_nonlinear_pde_snes():
    """Test Newton solver for a simple nonlinear PDE"""
    # Create mesh and function space
    mesh = dolfinx.generation.UnitSquareMesh(MPI.COMM_WORLD, 12, 15)
    V = fem.FunctionSpace(mesh, ("Lagrange", 1))
    u = fem.Function(V)
    v = TestFunction(V)
    F = inner(5.0, v) * dx - ufl.sqrt(u * u) * inner(
        grad(u), grad(v)) * dx - inner(u, v) * dx

    def boundary(x):
        """Define Dirichlet boundary (x = 0 or x = 1)."""
        return np.logical_or(x[0] < 1.0e-8, x[0] > 1.0 - 1.0e-8)

    u_bc = fem.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(1.0)
    bc = fem.DirichletBC(u_bc, fem.locate_dofs_geometrical(V, boundary))

    # Create nonlinear problem
    problem = NonlinearPDE_SNESProblem(F, u, bc)

    with u.vector.localForm() as u_local:
        u_local.set(0.9)
    b = dolfinx.cpp.la.create_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
    J = dolfinx.cpp.fem.create_matrix(problem.a_comp._cpp_object)

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
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.5)
    snes.solve(None, u.vector)
    assert snes.getConvergedReason() > 0
    assert snes.getIterationNumber() < 6
    # print(snes.getIterationNumber())
    # print(snes.getFunctionNorm())
