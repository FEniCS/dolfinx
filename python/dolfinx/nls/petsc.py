# Copyright (C) 2021-2025 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Methods for solving nonlinear equations using PETSc solvers."""

from __future__ import annotations

import typing
from enum import Enum

from mpi4py import MPI
from petsc4py import PETSc

if typing.TYPE_CHECKING:
    import dolfinx

    assert dolfinx.has_petsc4py

    from dolfinx.fem.problem import NonlinearProblem

import types

from dolfinx import cpp as _cpp
from dolfinx import fem
from dolfinx.fem.petsc import (
    create_matrix,
    create_matrix_block,
    create_matrix_nest,
    create_vector,
    create_vector_block,
    create_vector_nest,
)

__all__ = ["NewtonSolver", "SNESSolver", "SnesType"]


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


class setSNESFunctions(typing.Protocol):
    def F(self, snes: PETSc.SNES, x: PETSc.Vec, F: PETSc.Vec): ...  # type: ignore
    def J(self, snes: PETSc.SNES, x: PETSc.Vec, J: PETSc.Mat, P: PETSc.Mat): ...  # type: ignore


class SNESProblemProtocol(setSNESFunctions, typing.Protocol):
    @property
    def u(self) -> fem.Function | list[fem.Function]: ...

    @property
    def a(self) -> fem.Form | list[list[fem.Form]]: ...

    @property
    def L(self) -> fem.Form | list[fem.Form]: ...

    @property
    def P(self) -> fem.Form | list[list[fem.Form]] | None: ...

    @property
    def F(self) -> PETSc.Mat: ...  # type: ignore

    def copy_solution(self, x: PETSc.Vec): ...  # type: ignore

    def replace_solution(self, x: PETSc.Vec): ...  # type: ignore


class SnesType(Enum):
    default = 0
    block = 1
    nest = 2


def create_data_structures(
    a: typing.Union[list[list[fem.Form]], fem.Form],
    L: typing.Union[list[fem.Form], fem.Form],
    P: typing.Union[list[list[fem.Form]], list[fem.Form], fem.Form, None],
    snes_type: SnesType,
) -> tuple[PETSc.Mat, PETSc.Vec, PETSc.Vec, PETSc.Mat | None]:  # type: ignore
    """Create data-structures used in PETSc NEST solvers

    Args:
        a: The compiled bi-linear form(s)
        L: The compiled linear form(s)
        P: The compiled preconditioner form(s)
        snes_type: The type of NEST solver to use
    Returns:
        PETSc datastructures for the matrix A, vectors x and b, and preconditioner P
    """

    matrix_creator: typing.Union[None, typing.Callable[[PETSc.Mat], typing.Any]] = None  # type: ignore
    vector_creator: typing.Union[None, typing.Callable[[PETSc.Vec], typing.Any]] = None  # type: ignore
    if snes_type == SnesType.default:
        matrix_creator = create_matrix
        vector_creator = create_vector
    elif snes_type == SnesType.block:
        matrix_creator = create_matrix_block
        vector_creator = create_vector_block
    elif snes_type == SnesType.nest:
        matrix_creator = create_matrix_nest
        vector_creator = create_vector_nest
    else:
        raise ValueError("Unsupported SNES type")
    A = matrix_creator(a)  # type: ignore
    b = vector_creator(L)  # type: ignore
    x = vector_creator(L)  # type: ignore
    P = None if P is None else matrix_creator(P)  # type: ignore
    return A, x, b, P


class SNESSolver:
    def __init__(
        self, problem: SNESProblemProtocol, snes_type: SnesType, options: dict | None = None
    ):
        """Initialize a PETSc-SNES solver

        Args:
            problem: A problem instance for PETSc SNES
            options: Solver options. Can include any options for sub objects such as KSP and PC
        """
        self.problem = problem
        self.options = options if options is not None else {}
        self._A, self._x, self._b, self._P = create_data_structures(
            problem.a, problem.L, problem.P, snes_type
        )
        self.create_solver()
        self.error_if_not_converged = True

    def create_solver(self):
        """Create the PETSc SNES object and set solver options"""
        try:
            comm = self.problem.u.function_space.mesh.comm
        except AttributeError:
            comm = self.problem.u[0].function_space.mesh.comm
        self._snes = PETSc.SNES().create(comm=comm)
        opts = PETSc.Options()
        for key, v in self.options.items():
            opts[key] = v
        self._snes.setFromOptions()
        # Delete options from handler post setting
        for key, v in self.options.items():
            del opts[key]

    def solve(self) -> tuple[int, int]:
        """Solve the problem and update the solution in the problem instance

        Returns:
            Convergence reason and number of iterations
        """
        # Set function and Jacobian (in case the change in the SNES problem)
        self._snes.setFunction(self.problem.F, self._b)
        self._snes.setJacobian(self.problem.J, self._A, self._P)

        # Move current iterate into the work array.
        self.problem.copy_solution(self._x)

        # Solve problem
        self._snes.solve(None, self._x)
        # Check for convergence
        converged_reason = self._snes.getConvergedReason()
        if self.error_if_not_converged and converged_reason < 0:
            raise RuntimeError(f"Solver did not converge. Reason: {converged_reason}")

        # Update solution in problem
        self.problem.replace_solution(self._x)

        return converged_reason, self._snes.getIterationNumber()

    @property
    def krylov_solver(self):
        """Return the KSP object associated with the SNES object"""
        return self._snes.getKSP()

    def __del__(self):
        self._snes.destroy()
        self._A.destroy()
        self._b.destroy()
        self._x.destroy()
        if self._P is not None:
            self._P.destroy()
