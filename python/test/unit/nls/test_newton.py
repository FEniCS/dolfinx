# Copyright (C) 2018 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for Newton solver assembly"""

from mpi4py import MPI

import numpy as np
import pytest

import ufl
from dolfinx import cpp as _cpp
from dolfinx import default_real_type
from dolfinx.fem import Function, dirichletbc, form, functionspace, locate_dofs_geometrical
from dolfinx.mesh import create_unit_square
from ufl import TestFunction, TrialFunction, derivative, dx, grad, inner


class NonlinearPDEProblem:
    """Nonlinear problem class for a PDE problem."""

    def __init__(self, F, u, bc):
        V = u.function_space
        du = TrialFunction(V)
        self.L = form(F)
        self.a = form(derivative(F, u, du))
        self.bc = bc

    def form(self, x):
        from petsc4py import PETSc

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def F(self, x, b):
        """Assemble residual vector."""
        from petsc4py import PETSc

        from dolfinx.fem.petsc import apply_lifting, assemble_vector, set_bc

        with b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(b, self.L)
        apply_lifting(b, [self.a], bcs=[[self.bc]], x0=[x], alpha=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [self.bc], x, -1.0)

    def J(self, x, A):
        """Assemble Jacobian matrix."""
        from dolfinx.fem.petsc import assemble_matrix

        A.zeroEntries()
        assemble_matrix(A, self.a, bcs=[self.bc])
        A.assemble()

    def matrix(self):
        from dolfinx.fem.petsc import create_matrix

        return create_matrix(self.a)

    def vector(self):
        from dolfinx.fem.petsc import create_vector

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
        from petsc4py import PETSc

        from dolfinx.fem.petsc import apply_lifting, assemble_vector, set_bc

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        with F.localForm() as f_local:
            f_local.set(0.0)
        assemble_vector(F, self.L)
        apply_lifting(F, [self.a], bcs=[[self.bc]], x0=[x], alpha=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, [self.bc], x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        from dolfinx.fem.petsc import assemble_matrix

        J.zeroEntries()
        assemble_matrix(J, self.a, bcs=[self.bc])
        J.assemble()


@pytest.mark.petsc4py
class TestNLS:
    def test_linear_pde(self):
        """Test Newton solver for a linear PDE."""
        from petsc4py import PETSc

        # Create mesh and function space
        mesh = create_unit_square(MPI.COMM_WORLD, 12, 12)
        V = functionspace(mesh, ("Lagrange", 1))
        u = Function(V)
        v = TestFunction(V)
        F = inner(10.0, v) * dx - inner(grad(u), grad(v)) * dx

        bc = dirichletbc(
            PETSc.ScalarType(1.0),
            locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)),
            V,
        )

        # Create nonlinear problem
        problem = NonlinearPDEProblem(F, u, bc)

        def update(solver, dx, x):
            x.axpy(-1, dx)

        # Create Newton solver and solve
        solver = _cpp.nls.petsc.NewtonSolver(MPI.COMM_WORLD)
        solver.setF(problem.F, problem.vector())
        solver.setJ(problem.J, problem.matrix())
        solver.set_form(problem.form)
        solver.set_update(update)
        solver.atol = 1.0e-8
        solver.rtol = 1.0e2 * np.finfo(default_real_type).eps
        n, converged = solver.solve(u.x.petsc_vec)
        assert converged
        assert n == 1

        # Increment boundary condition and solve again
        bc.g.value[...] = PETSc.ScalarType(2.0)
        n, converged = solver.solve(u.x.petsc_vec)
        assert converged
        assert n == 1

        # Check reference counting
        ksp = solver.krylov_solver
        assert ksp.refcount == 2
        del solver
        assert ksp.refcount == 1

    def test_nonlinear_pde(self):
        """Test Newton solver for a simple nonlinear PDE"""
        from petsc4py import PETSc

        mesh = create_unit_square(MPI.COMM_WORLD, 12, 5)
        V = functionspace(mesh, ("Lagrange", 1))
        u = Function(V)
        v = TestFunction(V)
        F = inner(5.0, v) * dx - ufl.sqrt(u * u) * inner(grad(u), grad(v)) * dx - inner(u, v) * dx

        bc = dirichletbc(
            PETSc.ScalarType(1.0),
            locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)),
            V,
        )

        # Create nonlinear problem
        problem = NonlinearPDEProblem(F, u, bc)

        # Create Newton solver and solve
        u.x.array[:] = 0.9
        solver = _cpp.nls.petsc.NewtonSolver(MPI.COMM_WORLD)
        solver.setF(problem.F, problem.vector())
        solver.setJ(problem.J, problem.matrix())
        solver.set_form(problem.form)
        solver.atol = 1.0e-8
        solver.rtol = 1.0e2 * np.finfo(default_real_type).eps
        n, converged = solver.solve(u.x.petsc_vec)
        assert converged
        assert n < 6

        # Modify boundary condition and solve again
        bc.g.value[...] = 0.5
        n, converged = solver.solve(u.x.petsc_vec)
        assert converged
        assert n > 0 and n < 6

    def test_nonlinear_pde_snes(self):
        """Test Newton solver for a simple nonlinear PDE"""
        from petsc4py import PETSc

        from dolfinx.fem.petsc import create_matrix
        from dolfinx.la import create_petsc_vector

        mesh = create_unit_square(MPI.COMM_WORLD, 12, 15)
        V = functionspace(mesh, ("Lagrange", 1))
        u = Function(V)
        v = TestFunction(V)
        F = inner(5.0, v) * dx - ufl.sqrt(u * u) * inner(grad(u), grad(v)) * dx - inner(u, v) * dx

        u_bc = Function(V)
        u_bc.x.array[:] = 1.0
        bc = dirichletbc(
            u_bc,
            locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)),
        )

        # Create nonlinear problem
        problem = NonlinearPDE_SNESProblem(F, u, bc)

        u.x.array[:] = 0.9
        b = create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
        J = create_matrix(problem.a)

        # Create Newton solver and solve
        snes = PETSc.SNES().create()
        snes.setFunction(problem.F, b)
        snes.setJacobian(problem.J, J)

        snes.setTolerances(rtol=1.0e-9, max_it=10)
        snes.getKSP().setType("preonly")
        snes.getKSP().setTolerances(rtol=1.0e-9)
        snes.getKSP().getPC().setType("lu")

        x = u.x.petsc_vec.copy()
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        snes.solve(None, x)
        assert snes.getConvergedReason() > 0
        assert snes.getIterationNumber() < 6

        # Modify boundary condition and solve again
        u_bc.x.array[:] = 0.6
        snes.solve(None, x)
        assert snes.getConvergedReason() > 0
        assert snes.getIterationNumber() < 6
        # print(snes.getIterationNumber())
        # print(snes.getFunctionNorm())

        snes.destroy()
        b.destroy()
        J.destroy()
