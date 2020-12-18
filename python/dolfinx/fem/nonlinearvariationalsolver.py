# -*- coding: utf-8 -*-
# Copyright (C) 2020 Garth N. Wells and Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
import typing
import ufl
from petsc4py import PETSc
from ufl.equation import Equation
from dolfinx import fem
from dolfinx import cpp

___all__ = ["NonlinearVariationalSolver"]


class NonlinearPDEProblem(cpp.nls.NonlinearProblem):
    """Nonlinear problem class for a PDE problem."""

    def __init__(self, F, u, bcs, form_compiler_parameters={}):
        super().__init__()
        V = u.function_space
        du = ufl.TrialFunction(V)
        self.L = fem.Form(F.lhs, form_compiler_parameters=form_compiler_parameters)
        self.a = ufl.derivative(F.lhs, u, du)
        self.bcs = bcs
        self._F, self._J = None, None
        self.u = u

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
        fem.apply_lifting(self._F, [self.a], [self.bcs], [x], -1.0)
        self._F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(self._F, self.bcs, x, -1.0)

        return self._F

    def J(self, x):
        """Assemble Jacobian matrix."""
        if self._J is None:
            self._J = fem.assemble_matrix(self.a, self.bcs)
        else:
            self._J.zeroEntries()
            self._J = fem.assemble_matrix(self._J, self.a, self.bcs)
        self._J.assemble()
        return self._J


class NonlinearVariationalSolver():
    def __init__(self, variational_problem: Equation, u: fem.Function,
                 bcs: typing.List[fem.DirichletBC] = [], form_compiler_parameters={}):
        """
        Initialize the non-linear variational solver.
        By initialization we mean creating the underlying NonlinearVariationalProblem and Newton solver.

        .. code-block:: python

            solver = NonlinearVariationalSolver(F==0, u, [bc0, bc1], form_compiler_parameters={"optimize": True})

        """

        self.problem = NonlinearPDEProblem(variational_problem, u, bcs)
        self.newton_solver = cpp.nls.NewtonSolver(u.function_space.mesh.mpi_comm())

    def solve(self, petsc_options={}):
        n, converged = self.newton_solver.solve(self.problem, self.problem.u.vector)
        self.problem.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        if not converged:
            raise RuntimeError("Newton solver for non-linear variational problem did not converge.")
        return n, converged
