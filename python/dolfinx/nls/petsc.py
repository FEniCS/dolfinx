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

import ufl
import dolfinx.la
from dolfinx.fem.bcs import bcs_by_block as _bcs_by_block
from dolfinx.fem.forms import extract_function_spaces as _extract_spaces
from dolfinx.fem.petsc import set_bc
if typing.TYPE_CHECKING:
    import dolfinx

    assert dolfinx.has_petsc4py

    from dolfinx.fem.problem import NonlinearProblem

import types
from dolfinx.fem.forms import form as _create_form

from dolfinx import cpp as _cpp
from dolfinx import fem
from dolfinx.fem.petsc import (
    create_matrix,
    create_matrix_block,
    create_matrix_nest,
    create_vector,
    create_vector_block,
    create_vector_nest, apply_lifting,
    assemble_matrix_block,assemble_vector_block,assemble_matrix_nest,assemble_vector,
    assemble_matrix
)

__all__ = ["NewtonSolver", "SNESSolver", "SnesType", "create_snes_solver"]


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
        self, 
        J: PETSc.Mat,
        x: PETSc.Vec,
        b: PETSc.Vec,
        P: PETSc.Mat,
    ):
        """Initialize a PETSc-SNES solver

        Args:
            J: PETSc matrix that will hold the Jacobian
            x: PETSc vector that store the current iterate
            b: PETSc vector that will hold the right hand side
            P: PETSc matrix that will hold the preconditioner
            options: PETSc options to set for the SNES solver
        """
        self._A = J 
        self._b = b
        self._P = P
        self._x = x
        self._snes = PETSc.SNES().create(comm=self._A.comm)


    def set_options(self, options: dict):
        opts = PETSc.Options()
        for key, v in options.items():
            opts[key] = v
        self._snes.setFromOptions()
        # Delete options from handler post setting
        for key, _ in options.items():
            del opts[key]

    def solve(self) -> tuple[PETSc.Vec, int, int]:  # type: ignore
        """Solve the problem and update the solution in the problem instance.

        Returns:
            The solution, convergence reason and number of iterations
        """
        # Set function and Jacobian (in case the change in the SNES problem)
        self._snes.setFunction(self.F, self._b)
        self._snes.setJacobian(self.J, self._A, self._P)

        # Move current iterate into the work array.
        self.copy_solution(self._x)

        # Solve problem
        self._snes.solve(None, self._x)

        converged_reason = self._snes.getConvergedReason()
        return self._x, converged_reason, self._snes.getIterationNumber()

    @property
    def A(self) -> PETSc.Mat:  # type: ignore
        """Return the matrix associated with the SNES object"""
        return self._A

    @property
    def P(self) -> PETSc.Mat | None:  # type: ignore
        """Return the preconditioner matrix associated with the SNES object"""
        return self._P

    @property
    def x(self) -> PETSc.Vec:  # type: ignore
        """Return the solution vector associated with the SNES object"""
        return self._x

    @property
    def b(self) -> PETSc.Vec:  # type: ignore
        """Return the right hand side vector associated with the SNES object"""
        return self._b

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

    def set_F(self, F):
        self._F = F

    @property
    def F(self):
        return self._F

    def set_J(self, J):
        self._J = J

    @property
    def snes(self)->PETSc.SNES:  # type: ignore
        return self._snes

    @property
    def J(self):
        return self._J

    def set_copy_solution(self, copy_solution):
        self._copy_solution = copy_solution

    @property
    def copy_solution(self):
        return self._copy_solution
    

    def set_replace_solution(self, replace_solution):
        self._replace_solution = replace_solution

    @property
    def replace_solution(self):
        return self._replace_solution

def create_snes_solver(F: typing.Union[dolfinx.fem.Form, ufl.form.Form],
        u: dolfinx.fem.Function,
        J: typing.Optional[typing.Union[dolfinx.fem.Form, ufl.form.Form]] = None,
        P: typing.Optional[typing.Union[dolfinx.fem.Form, ufl.form.Form]] = None,
        snes_type: SnesType = SnesType.default,
        bcs: typing.Optional[list[dolfinx.fem.DirichletBC]] = None,
        form_compiler_options: typing.Optional[dict] = None,
        jit_options: typing.Optional[dict] = None,
        )->SNESSolver:

    # Compile all forms
    form_compiler_options = {} if form_compiler_options is None else form_compiler_options
    jit_options = {} if jit_options is None else jit_options
    residual = _create_form(
        F, form_compiler_options=form_compiler_options, jit_options=jit_options
    )
    if J is None:
        J = fem.forms.compute_jacobian(F, u)
    jacobian = _create_form(
        J, form_compiler_options=form_compiler_options, jit_options=jit_options
    )
    preconditioner = None
    if P is not None:
        preconditioner = _create_form(
            P, form_compiler_options=form_compiler_options, jit_options=jit_options
        )


    def replace_solution(x: PETSc.Vec):
        """Update the solution for the unknown `u` with the values in `x`.

        Args:
            x: Data that is inserted into the solution vector
        """
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(u.x.petsc_vec)
        u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def copy_solution(x: PETSc.Vec):
        """Copy the data in `u` into the vector `x`.

        Args:
            x: Vector to insert data into
        """
        u.x.petsc_vec.copy(x)

    def F(snes: PETSc.SNES, x: PETSc.Vec, F: PETSc.Vec):
        """Assemble the residual into the vector `F`.

        Args:
            snes: The solver instance
            x: The vector containing the point to evaluate the residual at.
            F: Vector to assemble the residual into
        """
        replace_solution(x)
        with F.localForm() as f_local:
            f_local.set(0.0)
        assemble_vector(F, residual)
        apply_lifting(F, [jacobian], bcs=[bcs], x0=[x], alpha=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, bcs, x, -1.0)

    def J(snes, x, J, P):
        """Assemble the Jacobian matrix and preconditioner.

        Args:
            snes: The solver instance
            x: The vector containing the point to evaluate at
            J: Matrix to assemble the Jacobian into
            P: Matrix to assemble the preconditioner into
        """
        # Copy existing soultion into the function used in the residual and Jacobian
        replace_solution(x)

        # Assemble Jacobian
        J.zeroEntries()
        assemble_matrix(J, jacobian, bcs, diagonal=1.0)
        J.assemble()

        if preconditioner is not None:
            P.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix(P, preconditioner, bcs, diagonal=1.0)
            P.assemble()


    def F_block(snes: PETSc.SNES, x: PETSc.Vec, F: PETSc.Vec):
        """Assemble the residual into the vector F.

        Args:
            snes: The solver instance
            x: The vector containing the latest solution
            F: Vector to assemble the residual into
        """
        assert x.getType() != "nest", "Vector x should be non-nested"
        assert F.getType() != "nest", "Vector F should be non-nested"
        with F.localForm() as f_local:
            f_local.set(0.0)

        replace_solution_block(x)
        assemble_vector_block(F, residual, jacobian, bcs=bcs, x0=x, alpha=-1.0)

    def J_block(snes, x, J, P):
        """Assemble the Jacobian matrix.

        Args:
            snes: The solver instance
            x: The vector containing the latest solution
            J: Matrix to assemble the Jacobian into
            P: Matrix to assemble the preconditioner into
        """
        replace_solution_block(x)
        assert x.getType() != "nest", "Vector x should be non-nested"
        assert J.getType() != "nest", "Matrix J should be non-nested"
        assert P.getType() != "nest", "Matrix P should be non-nested"
        J.zeroEntries()
        assemble_matrix_block(J, jacobian, bcs=bcs, diagonal=1.0)
        J.assemble()
        if preconditioner is not None:
            P.zeroEntries()
            assemble_matrix_block(P, preconditioner, bcs=bcs, diagonal=1.0)
            P.assemble()

    def replace_solution_block(x: PETSc.Vec):
        """Update the solution for the unknown `u` with the values in `x`.

        Args:
            x: Data that is inserted into the solution vector
        """
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        offset_start = 0
        for ui in u:
            Vi = ui.function_space
            num_sub_dofs = Vi.dofmap.index_map.size_local * Vi.dofmap.index_map_bs
            ui.x.petsc_vec.array_w[:num_sub_dofs] = x.array_r[
                offset_start : offset_start + num_sub_dofs
            ]
            ui.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            offset_start += num_sub_dofs

    def copy_solution_block(x: PETSc.Vec):
        """Copy the data in `u` into the vector `x`.

        Args:
            x: Vector to insert data into
        """
        u_petsc = [ui.x.petsc_vec.array_r for ui in u]
        index_maps = [
            (ui.function_space.dofmap.index_map, ui.function_space.dofmap.index_map_bs)
            for ui in u
        ]
        _cpp.la.petsc.scatter_local_vectors(
            x,
            u_petsc,
            index_maps,
        )
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


    def F_nest(snes: PETSc.SNES, x: PETSc.Vec, F: PETSc.Vec):
        """Assemble the residual into vector `F`.

        Args:
            snes: The solver instance
            x: The vector containing the latest solution
            F: Vector to assemble the residual into
        """
        assert x.getType() == "nest" and F.getType() == "nest"
        replace_solution_nest(x)
        bcs1 = _bcs_by_block(_extract_spaces(jacobian, 1), bcs)
        sub_vectors = x.getNestSubVecs()
        for L, F_sub, a in zip(residual, F.getNestSubVecs(), jacobian):
            with F_sub.localForm() as F_sub_local:
                F_sub_local.set(0.0)
            assemble_vector(F_sub, L)
            apply_lifting(F_sub, a, bcs=bcs1, x0=sub_vectors, alpha=-1.0)
            F_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Set bc value in RHS
        bcs0 = _bcs_by_block(_extract_spaces(residual), bcs)
        for F_sub, bc, x_sub in zip(F.getNestSubVecs(), bcs0, sub_vectors):
            set_bc(F_sub, bc, x_sub, -1.0)

        # Must assemble F here in the case of nest matrices
        F.assemble()

    def J_nest(snes, x, J, P):
        """Assemble the Jacobian matrix.

        Args:
            snes: The solver instance
            x: The vector containing the latest solution
            J: Matrix to assemble the Jacobian into
            P: Matrix to assemble the preconditioner into
        """
        replace_solution_nest(x)
        assert J.getType() == "nest" and P.getType() == "nest"
        J.zeroEntries()
        assemble_matrix_nest(J, jacobian, bcs=bcs, diagonal=1.0)
        J.assemble()
        
        if preconditioner is not None:
            P.zeroEntries()
            assemble_matrix_nest(P, preconditioner, bcs=bcs, diagonal=1.0)
            P.assemble()

    def replace_solution_nest(x: PETSc.Vec):
        """Update the solution for the unknown `u` with the values in `x`.

        Args:
            x: Data that is inserted into the solution vector
        """
        subvecs = x.getNestSubVecs()
        for x_sub, var_sub in zip(subvecs, u):
            x_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            with x_sub.localForm() as _x:
                var_sub.x.array[:] = _x.array_r

    def copy_solution_nest(x: PETSc.Vec):
        """Copy the data in `u` into the vector `x`.

        Args:
            x: Vector to insert data into
        """
        wrapped_sol = [dolfinx.la.create_petsc_vector_wrap(u_i.x) for u_i in u]
        u_nest = PETSc.Vec().createNest(wrapped_sol)
        u_nest.copy(x)
        u_nest.destroy()
        [wrapped.destroy() for wrapped in wrapped_sol]

    A,x, b, P = create_data_structures(jacobian, residual, preconditioner, snes_type)
    snes_solver = SNESSolver(A,x, b, P)
    if snes_type == SnesType.default:
        snes_solver.set_F(F)
        snes_solver.set_J(J)
        snes_solver.set_copy_solution(copy_solution)
        snes_solver.set_replace_solution(replace_solution)
    elif snes_type == SnesType.block:
        snes_solver.set_F(F_block)
        snes_solver.set_J(J_block)
        snes_solver.set_copy_solution(copy_solution_block)
        snes_solver.set_replace_solution(replace_solution_block)
    elif snes_type == SnesType.nest:
        snes_solver.set_F(F_nest)
        snes_solver.set_J(J_nest)
        snes_solver.set_copy_solution(copy_solution_nest)
        snes_solver.set_replace_solution(replace_solution_nest)
    else:
        raise ValueError(f"Unsupported SNES type {snes_type}")
    return snes_solver