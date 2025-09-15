# Copyright (C) 2021-2025 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""(Deprecated) Methods for solving nonlinear equations using PETSc
solvers."""

from __future__ import annotations

import typing
import warnings

from mpi4py import MPI
from petsc4py import PETSc

if typing.TYPE_CHECKING:
    import dolfinx

    assert dolfinx.has_petsc4py

    from dolfinx.fem.petsc import NewtonSolverNonlinearProblem

import types

from dolfinx import cpp as _cpp
from dolfinx import fem
from dolfinx.fem.petsc import (
    create_matrix,
    create_vector,
)

__all__ = ["NewtonSolver"]


class NewtonSolver(_cpp.nls.petsc.NewtonSolver):
    def __init__(self, comm: MPI.Intracomm, problem: NewtonSolverNonlinearProblem):
        """(Deprecated) A Newton solver for non-linear problems.

        Note:
            This class is deprecated in favour of
            :class:`dolfinx.fem.petsc.NonlinearProblem`, a high
            level interface to PETSc SNES.
        """
        super().__init__(comm)

        warnings.warn(
            (
                "dolfinx.nls.petsc.NewtonSolver is deprecated. "
                + "Use dolfinx.fem.petsc.NonlinearProblem, "
                + "a high level interface to PETSc SNES."
            ),
            DeprecationWarning,
        )

        # Create matrix and vector to be used for assembly
        # of the non-linear problem
        self._A = create_matrix(problem.a)
        self.setJ(problem.J, self._A)
        self._b = create_vector(problem.L)
        self.setF(problem.F, self._b)
        self.set_form(problem.form)

    def __del__(self):
        self._A.destroy()
        self._b.destroy()

    def solve(self, u: fem.Function):
        """Solve non-linear problem into function u. Returns the number
        of iterations and if the solver converged."""
        n, converged = super().solve(u.x.petsc_vec)
        u.x.scatter_forward()
        return n, converged

    @property
    def A(self) -> PETSc.Mat:  # type: ignore
        """Jacobian matrix"""
        return self._A

    @property
    def b(self) -> PETSc.Vec:  # type: ignore
        """Residual vector"""
        return self._b

    def setP(self, P: types.FunctionType, Pmat: PETSc.Mat):  # type: ignore
        """
        Set the function for computing the preconditioner matrix

        Args:
            P: Function to compute the preconditioner matrix
            Pmat: Matrix to assemble the preconditioner into

        """
        super().setP(P, Pmat)
