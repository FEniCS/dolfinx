# Copyright (C) 2018 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for Newton solver assembly."""

from mpi4py import MPI

import numpy as np
import pytest

import ufl
from dolfinx.fem import Function, dirichletbc, form, functionspace, locate_dofs_geometrical
from dolfinx.mesh import create_unit_square
from ufl import TestFunction, TrialFunction, derivative, dx, grad, inner


class NonlinearPDE_SNESProblem:
    """Nonlinear problem class for a PDE problem using SNES interface."""

    def __init__(self, F, u, bc):
        """Initialize nonlinear PDE problem."""
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
        from petsc4py import PETSc

        from dolfinx.fem.petsc import assemble_matrix

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        J.zeroEntries()
        assemble_matrix(J, self.a, bcs=[self.bc])
        J.assemble()


@pytest.mark.petsc4py
class TestNLS:
    """Test SNES nonlinear solver for PDEs."""

    def test_nonlinear_pde_snes(self):
        """Test SNES solver for a simple nonlinear PDE."""
        from petsc4py import PETSc

        from dolfinx.fem.petsc import create_matrix, create_vector

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
        b = create_vector(V)
        J = create_matrix(problem.a)

        # Create Newton solver and solve
        snes = PETSc.SNES().create()
        snes.setFunction(problem.F, b)
        snes.setJacobian(problem.J, J)

        snes.setTolerances(rtol=1.0e-9, max_it=10)
        snes.getKSP().setType("preonly")
        snes.getKSP().setTolerances(rtol=1.0e-9)
        snes.getKSP().getPC().setType("lu")

        # For SNES line search to function correctly it is necessary that the
        # u.x.petsc_vec in the Jacobian and residual is *not* passed to
        # snes.solve.
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
