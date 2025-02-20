# Copyright (C) 2021-2025 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Methods for solving nonlinear equations using PETSc solvers."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from mpi4py import MPI
    from petsc4py import PETSc

    import dolfinx

    assert dolfinx.has_petsc4py

    from dolfinx.fem.problem import NonlinearProblem

import types

from dolfinx import cpp as _cpp
from dolfinx import fem
from dolfinx.fem.petsc import create_matrix, create_vector

__all__ = ["NewtonSolver", "SNESSolver"]


class NewtonSolver(_cpp.nls.petsc.NewtonSolver):
    def __init__(self, comm: MPI.Intracomm, problem: NonlinearProblem):
        """A Newton solver for non-linear problems."""
        super().__init__(comm)

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

class SNESSolver:
    def __init__(self, problem: dolfinx.fem.petsc.SNESProblem, options: dict|None = None):
        """Initialize a PETSc-SNES solver

        Args:
            problem: A problem instance for PETSc SNES
            options: Solver options. Can include any options for sub objects such as KSP and PC
        """
        self.problem = problem
        self.options = options if options is not None else {}
        self.create_solver()
        self.create_data_structures()
        self.error_if_not_converged = True

    def create_solver(self):
        """Create the PETSc SNES object and set solver options"""
        self._snes = PETSc.SNES().create(comm=self.problem.u.function_space.mesh.comm)
        self._snes.setOptionsPrefix("snes_solve")
        option_prefix = self._snes.getOptionsPrefix()
        opts = PETSc.Options()
        opts.prefixPush(option_prefix)
        for key, v in self.options.items():
            opts[key] = v
        opts.prefixPop()
        self._snes.setFromOptions()

    def create_data_structures(self):
        """
        Create PETSc objects for the matrix, residual and solution
        """
        self._A = dolfinx.fem.petsc.create_matrix(self.problem.a)
        self._b = dolfinx.fem.Function(self.problem.u.function_space, name="Residual")
        self._x = dolfinx.fem.Function(self.problem.u.function_space, name="work-array")

    @property
    def _b_petsc(self):
        return self._b.x.petsc_vec
    
    def solve(self):
        # Set function and Jacobian (in case the change in the SNES problem)
        self._snes.setFunction(self.problem.F, self._b_petsc)
        self._snes.setJacobian(self.problem.J, self._A)

        # Move current iterate into the work array.
        self._x.interpolate(self.problem.u)
        self._snes.solve(None, self._x.x.petsc_vec)
        self._x.x.scatter_forward()

        # Check for convergence
        converged_reason = self._snes.getConvergedReason()
        if self.error_if_not_converged and converged_reason < 0:
            raise RuntimeError(f"Solver did not converge. Reason: {converged_reason}")

        self.problem.u.x.array[:] = self._x.x.array 
        return converged_reason, self._snes.getIterationNumber()

    def __del__(self):
        self._snes.destroy()
        self._A.destroy()

