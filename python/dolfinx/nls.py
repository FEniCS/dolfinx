# Copyright (C) 2021 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Methods for solving nonlinear equations."""

import types

from dolfinx import cpp as _cpp
from dolfinx import fem

import mpi4py
from petsc4py import PETSc


class NewtonSolver(_cpp.nls.NewtonSolver):
    def __init__(self, comm: mpi4py.MPI.Intracomm, problem: fem.NonlinearProblem):
        """
        Create a Newton solver for a given MPI communicator and non-linear problem.
        """
        super().__init__(comm)

        # Create matrix and vector to be used for assembly
        # of the non-linear problem
        self._A = fem.create_matrix(problem.a)
        self.setJ(problem.J, self._A)
        self._b = fem.create_vector(problem.L)
        self.setF(problem.F, self._b)
        self.set_form(problem.form)

    def solve(self, u: fem.Function):
        """
        Solve non-linear problem into function u.
        Returns the number of iterations and if the solver converged
        """
        n, converged = super().solve(u.vector())
        u.x.scatter_forward()
        return n, converged

    @property
    def A(self) -> PETSc.Mat:
        """Get the Jacobian matrix"""
        return self._A

    @property
    def b(self) -> PETSc.Vec:
        """Get the residual vector"""
        return self._b

    def setP(self, P: types.FunctionType, Pmat: PETSc.Mat):
        """
        Set the function for computing the preconditioner matrix
        Parameters
        -----------
        P
          Function to compute the preconditioner matrix b (x, P)
        Pmat
          The matrix to assemble the preconditioner into
        """
        super().setP(P, Pmat)
