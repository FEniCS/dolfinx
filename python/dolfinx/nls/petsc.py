# Copyright (C) 2021 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Methods for solving nonlinear equations using PETSc solvers."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from dolfinx.fem.problem import NonlinearProblem
    from mpi4py import MPI
    from petsc4py import PETSc

import types

from dolfinx import cpp as _cpp
from dolfinx import fem

__all__ = ["NewtonSolver"]


class NewtonSolver(_cpp.nls.petsc.NewtonSolver):
    def __init__(self, comm: MPI.Intracomm, problem: NonlinearProblem):
        """A Newton solver for non-linear problems."""
        super().__init__(comm)

        # Create matrix and vector to be used for assembly
        # of the non-linear problem
        self._A = fem.petsc.create_matrix(problem.a)
        self.setJ(problem.J, self._A)
        self._b = fem.petsc.create_vector(problem.L)
        self.setF(problem.F, self._b)
        self.set_form(problem.form)

    def solve(self, u: fem.Function):
        """Solve non-linear problem into function u. Returns the number
        of iterations and if the solver converged."""
        n, converged = super().solve(u.vector)
        u.x.scatter_forward()
        return n, converged

    @property
    def A(self) -> PETSc.Mat:
        """Jacobian matrix"""
        return self._A

    @property
    def b(self) -> PETSc.Vec:
        """Residual vector"""
        return self._b

    def setP(self, P: types.FunctionType, Pmat: PETSc.Mat):
        """
        Set the function for computing the preconditioner matrix

        Args:
            P: Function to compute the preconditioner matrix
            Pmat: Matrix to assemble the preconditioner into
        """
        super().setP(P, Pmat)
