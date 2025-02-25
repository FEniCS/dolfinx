# Copyright (C) 2021-2025 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Methods for solving nonlinear equations using PETSc solvers."""

from __future__ import annotations

import typing
from functools import partial

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.la
import ufl
from dolfinx.fem.bcs import bcs_by_block as _bcs_by_block
from dolfinx.fem.forms import extract_function_spaces as _extract_spaces

if typing.TYPE_CHECKING:
    import dolfinx

    assert dolfinx.has_petsc4py

    from dolfinx.fem.problem import NonlinearProblem

import types

from dolfinx import cpp as _cpp
from dolfinx import fem
from dolfinx.fem.forms import form as _create_form
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix_block,
    assemble_matrix_nest,
    assemble_vector,
    assemble_vector_block,
    copy_block_vec_to_functions,
    copy_function_to_vec,
    copy_functions_to_block_vec,
    copy_functions_to_nest_vec,
    copy_nest_vec_to_functions,
    copy_vec_to_function,
    create_matrix,
    create_matrix_block,
    create_matrix_nest,
    create_vector,
    create_vector_block,
    create_vector_nest,
    set_bc,
)

__all__ = ["NewtonSolver", "SNESSolver", "create_snes_solver"]


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


def create_snes_matrices_and_vectors(
    a: typing.Union[list[list[fem.Form]], fem.Form],
    L: typing.Union[list[fem.Form], fem.Form],
    P: typing.Union[list[list[fem.Form]], list[fem.Form], fem.Form, None],
    assembly_type: fem.petsc.AssemblyType,
) -> tuple[  # type: ignore
    PETSc.Mat,
    PETSc.Vec,
    PETSc.Mat | None,
    PETSc.Vec,
]:
    """Create data-structures used in PETSc NEST solvers.

    Args:
        a: The compiled bi-linear form(s), the Jacobian.
        L: The compiled linear form(s), the residual.
        P: The compiled preconditioner form(s).
        assembly_type: The type of matrix to use.
    Returns:
        PETSc datastructures for the matrix A, b, and preconditioner P relating to the input,
        as well as a work-vector `x`.
    """

    matrix_creator: typing.Union[None, typing.Callable[[PETSc.Mat], typing.Any]] = None  # type: ignore
    vector_creator: typing.Union[None, typing.Callable[[PETSc.Vec], typing.Any]] = None  # type: ignore
    if assembly_type == fem.petsc.AssemblyType.default:
        matrix_creator = create_matrix
        vector_creator = create_vector
    elif assembly_type == fem.petsc.AssemblyType.block:
        matrix_creator = create_matrix_block
        vector_creator = create_vector_block
    elif assembly_type == fem.petsc.AssemblyType.nest:
        matrix_creator = create_matrix_nest
        vector_creator = create_vector_nest
    else:
        raise ValueError("Unsupported SNES type")
    A = matrix_creator(a)  # type: ignore
    b = vector_creator(L)  # type: ignore
    x = vector_creator(L)  # type: ignore
    P = None if P is None else matrix_creator(P)  # type: ignore
    return A, b, P, x


class SNESSolver:
    def __init__(
        self,
        F: typing.Union[dolfinx.fem.Form, ufl.form.Form],
        u: dolfinx.fem.Function,
        bcs: typing.Optional[list[dolfinx.fem.DirichletBC]] = None,
        J: typing.Optional[typing.Union[dolfinx.fem.Form, ufl.form.Form]] = None,
        P: typing.Optional[typing.Union[dolfinx.fem.Form, ufl.form.Form]] = None,
        assembly_type: fem.petsc.AssemblyType = fem.petsc.AssemblyType.default,
        form_compiler_options: typing.Optional[dict] = None,
        jit_options: typing.Optional[dict] = None,
        snes_options: typing.Optional[dict] = None,
    ):
        """Class for interfacing with SNES.

        Solves problems of the form :math:`F_i(u, v) = 0, i=0,...N\\ \\forall v \\in V`
        where :math:`u=(u_0,...,u_N), v=(v_0,...,v_N)` using PETSc SNES as the non-linear solver.

        Args:
            F: List of PDE residuals :math:`F_i`.
            u: List of unknowns of the block system.
            bcs: List of Dirichlet boundary conditions.
            J: Rectangular array of bi-linear forms representing the
                Jacobian :math:`J_ij = dF_i/du_j`.
            P: Rectangular array of bi-linear forms representing the preconditioner.
            assembly_type: Determines what kind of matrix the problem is assembled into.
                If `N=0`, i.e one is solving for a single form, the assembly type is `default`.
                For blocked, the assembly type is `block` or `nest`.
            form_compiler_options: Options used in FFCx
                compilation of this form. Run ``ffcx --help`` at the
                command line to see all available options.
            jit_options: Options used in CFFI JIT compilation of C
                code generated by FFCx. See ``python/dolfinx/jit.py``
                for all available options. Takes priority over all other
                option values.
            snes_options: Options to pass to the PETSc SNES object.
        """

        self._u = u
        self._snes, self._x = create_snes_solver(
            F, self._u, bcs, J, P, assembly_type, form_compiler_options, jit_options
        )
        if assembly_type == fem.petsc.AssemblyType.default:
            self._copy_function_to_vec = copy_function_to_vec
            self._copy_vec_to_function = copy_vec_to_function
        elif assembly_type == fem.petsc.AssemblyType.block:
            self._copy_function_to_vec = copy_functions_to_block_vec
            self._copy_vec_to_function = copy_block_vec_to_functions
        elif assembly_type == fem.petsc.AssemblyType.nest:
            self._copy_function_to_vec = copy_functions_to_nest_vec
            self._copy_vec_to_function = copy_nest_vec_to_functions
        else:
            raise ValueError("Unsupported Assembly type")

        snes_options = {} if snes_options is None else snes_options
        opts = PETSc.Options()  # type: ignore
        for key, v in snes_options.items():
            opts[key] = v
        self._snes.setFromOptions()

        # Delete options from handler post setting
        for key, _ in snes_options.items():
            del opts[key]

    def solve(self) -> tuple[PETSc.Vec, int, int]:  # type: ignore
        """Solve the problem and update the solution in the problem instance.

        Returns:
            The solution, convergence reason and number of iterations
        """

        # Move current iterate into the work array.
        self.copy_function_to_vec(self._u, self._x)

        # Solve problem
        self._snes.solve(None, self._x)

        converged_reason = self._snes.getConvergedReason()
        return self._x, converged_reason, self._snes.getIterationNumber()

    def __del__(self):
        self._snes.destroy()
        self._x.destroy()

    @property
    def snes(self) -> PETSc.SNES:  # type: ignore
        return self._snes

    @property
    def copy_function_to_vec(self):
        return self._copy_function_to_vec

    @property
    def copy_vec_to_function(self):
        return self._copy_vec_to_function


def F_default(
    u: dolfinx.fem.Function,
    residual: dolfinx.fem.Form,
    jacobian: dolfinx.fem.Form,
    bcs: list[dolfinx.fem.DirichletBC],
    snes: PETSc.SNES,  # type: ignore
    x: PETSc.Vec,  # type: ignore
    F: PETSc.Vec,  # type: ignore
):
    """Assemble the residual into the vector `F`.

    Args:
        u: Function tied to the solution vector within the residual and Jacobian.
        residual: Form of the residual.
        jacobian: Form of the Jacobian.
        bcs: List of Dirichlet boundary conditions.
        snes: The solver instance.
        x: The vector containing the point to evaluate the residual at.
        F: Vector to assemble the residual into.
    """
    copy_vec_to_function(u, x)
    with F.localForm() as f_local:
        f_local.set(0.0)
    assemble_vector(F, residual)
    apply_lifting(F, [jacobian], bcs=[bcs], x0=[x], alpha=-1.0)
    F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
    set_bc(F, bcs, x, -1.0)


def J_default(
    u: dolfinx.fem.Function,
    jacobian: dolfinx.fem.Form,
    preconditioner: typing.Optional[dolfinx.fem.Form],
    bcs: list[dolfinx.fem.DirichletBC],
    snes: PETSc.SNES,  # type: ignore
    x: PETSc.Vec,  # type: ignore
    J: PETSc.Mat,  # type: ignore
    P: PETSc.Mat,  # type: ignore
):
    """Assemble the Jacobian matrix and preconditioner.

    Args:
        u: Function tied to the solution vector within the residual and jacobian
        jacobian: Form of the Jacobian
        preconditioner: Form of the preconditioner
        bcs: List of Dirichlet boundary conditions
        snes: The solver instance
        x: The vector containing the point to evaluate at
        J: Matrix to assemble the Jacobian into
        P: Matrix to assemble the preconditioner into
    """
    # Copy existing soultion into the function used in the residual and Jacobian
    copy_vec_to_function(u, x)

    # Assemble Jacobian
    J.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(J, jacobian, bcs, diagonal=1.0)  # type: ignore
    J.assemble()

    if preconditioner is not None:
        P.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(P, preconditioner, bcs, diagonal=1.0)  # type: ignore
        P.assemble()


def F_block(
    u: list[dolfinx.fem.Function],
    residual: list[dolfinx.fem.Form],
    jacobian: list[list[dolfinx.fem.Form]],
    bcs: list[dolfinx.fem.DirichletBC],
    snes: PETSc.SNES,  # type: ignore
    x: PETSc.Vec,  # type: ignore
    F: PETSc.Vec,  # type: ignore
):
    """Assemble the residual into the vector F.

    Args:
        u: List of functions tied to the solution vector within the residual and jacobian
        residual: List of forms of the residual
        jacobian: List of list of forms of the Jacobian
        bcs: List of Dirichlet boundary conditions
        snes: The solver instance
        x: The vector containing the latest solution
        F: Vector to assemble the residual into
    """
    assert x.getType() != "nest", "Vector x should be non-nested"
    assert F.getType() != "nest", "Vector F should be non-nested"
    with F.localForm() as f_local:
        f_local.set(0.0)

    copy_block_vec_to_functions(u, x)
    assemble_vector_block(F, residual, jacobian, bcs=bcs, x0=x, alpha=-1.0)  # type: ignore


def J_block(
    u: list[dolfinx.fem.Function],
    jacobian: list[list[dolfinx.fem.Form]],
    preconditioner: typing.Optional[list[dolfinx.fem.Form]],
    bcs: list[dolfinx.fem.DirichletBC],
    snes: PETSc.SNES,  # type: ignore
    x: PETSc.Vec,  # type: ignore
    J: PETSc.Mat,  # type: ignore
    P: PETSc.Mat,  # type: ignore
):
    """Assemble the Jacobian matrix.

    Args:
        u: List of functions tied to the solution vector within the residual and jacobian
        jacobian: List of list of forms of the Jacobian
        preconditioner: List of forms of the preconditioner
        bcs: List of Dirichlet boundary conditions
        snes: The solver instance
        x: The vector containing the latest solution
        J: Matrix to assemble the Jacobian into
        P: Matrix to assemble the preconditioner into
    """
    copy_block_vec_to_functions(u, x)
    assert x.getType() != "nest", "Vector x should be non-nested"
    assert J.getType() != "nest", "Matrix J should be non-nested"
    assert P.getType() != "nest", "Matrix P should be non-nested"
    J.zeroEntries()
    assemble_matrix_block(J, jacobian, bcs=bcs, diagonal=1.0)  # type: ignore
    J.assemble()
    if preconditioner is not None:
        P.zeroEntries()
        assemble_matrix_block(P, preconditioner, bcs=bcs, diagonal=1.0)  # type: ignore
        P.assemble()


def F_nest(
    u: list[dolfinx.fem.Function],
    residual: list[dolfinx.fem.Form],
    jacobian: list[list[dolfinx.fem.Form]],
    bcs: list[dolfinx.fem.DirichletBC],
    snes: PETSc.SNES,  # type: ignore
    x: PETSc.Vec,  # type: ignore
    F: PETSc.Vec,  # type: ignore
):
    """Assemble the residual into the vector F.

    Args:
        u: List of functions tied to the solution vector within the residual and jacobian
        residual: List of forms of the residual
        jacobian: List of list of forms of the Jacobian
        bcs: List of Dirichlet boundary conditions
        snes: The solver instance
        x: The vector containing the latest solution
        F: Vector to assemble the residual into
    """
    assert x.getType() == "nest" and F.getType() == "nest"
    copy_nest_vec_to_functions(u, x)
    bcs1 = _bcs_by_block(_extract_spaces(jacobian, 1), bcs)
    sub_vectors = x.getNestSubVecs()
    for L, F_sub, a in zip(residual, F.getNestSubVecs(), jacobian):
        with F_sub.localForm() as F_sub_local:
            F_sub_local.set(0.0)
        assemble_vector(F_sub, L)
        apply_lifting(F_sub, a, bcs=bcs1, x0=sub_vectors, alpha=-1.0)
        F_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore

    # Set bc value in RHS
    bcs0 = _bcs_by_block(_extract_spaces(residual), bcs)
    for F_sub, bc, x_sub in zip(F.getNestSubVecs(), bcs0, sub_vectors):
        set_bc(F_sub, bc, x_sub, -1.0)

    # Must assemble F here in the case of nest matrices
    F.assemble()


def J_nest(
    u: list[dolfinx.fem.Function],
    jacobian: list[list[dolfinx.fem.Form]],
    preconditioner: typing.Optional[list[dolfinx.fem.Form]],
    bcs: list[dolfinx.fem.DirichletBC],
    snes: PETSc.SNES,  # type: ignore
    x: PETSc.Vec,  # type: ignore
    J: PETSc.Mat,  # type: ignore
    P: PETSc.Mat,  # type: ignore
):
    """Assemble the Jacobian matrix.

    Args:
        u: List of functions tied to the solution vector within the residual and jacobian
        jacobian: List of list of forms of the Jacobian
        preconditioner: List of forms of the preconditioner
        bcs: List of Dirichlet boundary conditions
        snes: The solver instance
        x: The vector containing the latest solution
        J: Matrix to assemble the Jacobian into
        P: Matrix to assemble the preconditioner into
    """
    copy_nest_vec_to_functions(u, x)
    assert J.getType() == "nest" and P.getType() == "nest"
    J.zeroEntries()
    assemble_matrix_nest(J, jacobian, bcs=bcs, diagonal=1.0)  # type: ignore
    J.assemble()

    if preconditioner is not None:
        P.zeroEntries()
        assemble_matrix_nest(P, preconditioner, bcs=bcs, diagonal=1.0)  # type: ignore
        P.assemble()


def create_snes_solver(
    F: typing.Union[dolfinx.fem.Form, ufl.form.Form],
    u: dolfinx.fem.Function,
    bcs: typing.Optional[list[dolfinx.fem.DirichletBC]] = None,
    J: typing.Optional[typing.Union[dolfinx.fem.Form, ufl.form.Form]] = None,
    P: typing.Optional[typing.Union[dolfinx.fem.Form, ufl.form.Form]] = None,
    assembly_type: fem.petsc.AssemblyType = fem.petsc.AssemblyType.default,
    form_compiler_options: typing.Optional[dict] = None,
    jit_options: typing.Optional[dict] = None,
) -> tuple[PETSc.SNES, PETSc.Vec]:  # type: ignore
    """Create a PETSc SNES solver instance and a vector to store solutions in.

    Solves problems of the form :math:`F_i(u, v) = 0, i=0,...N\\ \\forall v \\in V`
    where :math:`u=(u_0,...,u_N), v=(v_0,...,v_N)` using PETSc SNES as the non-linear solver.

    Note:
        The user is responsible for the destruction of the returned objects.

    Args:
        F: List of PDE residuals :math:`F_i`.
        u: List of unknowns of the block system.
        bcs: List of Dirichlet boundary conditions.
        J: Rectangular array of bi-linear forms representing the
            Jacobian :math:`J_ij = dF_i/du_j`.
        P: Rectangular array of bi-linear forms representing the preconditioner.
        assembly_type: Determines what kind of matrix the problem is assembled into.
            If `N=0`, i.e one is solving for a single form, the assembly type is `default`.
            For blocked, the assembly type is `block` or `nest`.
        form_compiler_options: Options used in FFCx
            compilation of this form. Run ``ffcx --help`` at the
            command line to see all available options.
        jit_options: Options used in CFFI JIT compilation of C
            code generated by FFCx. See ``python/dolfinx/jit.py``
            for all available options. Takes priority over all other
            option values.
    Returns:
        A PETSc SNES solver instance and a vector to store solutions in.
    """
    # Compile all forms
    bcs = [] if bcs is None else bcs
    form_compiler_options = {} if form_compiler_options is None else form_compiler_options
    jit_options = {} if jit_options is None else jit_options

    # Compile forms
    residual = _create_form(F, form_compiler_options=form_compiler_options, jit_options=jit_options)
    if J is None:
        J = fem.forms.compute_jacobian(F, u)
    jacobian = _create_form(J, form_compiler_options=form_compiler_options, jit_options=jit_options)
    preconditioner = None
    if P is not None:
        preconditioner = _create_form(
            P, form_compiler_options=form_compiler_options, jit_options=jit_options
        )

    A, b, P, x = create_snes_matrices_and_vectors(jacobian, residual, preconditioner, assembly_type)
    snes = PETSc.SNES().create(comm=A.comm)  # type: ignore

    # Set function and Jacobian
    if assembly_type == fem.petsc.AssemblyType.default:
        snes.setFunction(partial(F_default, u, residual, jacobian, bcs), b)
        snes.setJacobian(partial(J_default, u, jacobian, preconditioner, bcs), A, P)
    elif assembly_type == fem.petsc.AssemblyType.block:
        snes.setFunction(partial(F_block, u, residual, jacobian, bcs), b)
        snes.setJacobian(partial(J_block, u, jacobian, preconditioner, bcs), A, P)
    elif assembly_type == fem.petsc.AssemblyType.nest:
        snes.setFunction(partial(F_nest, u, residual, jacobian, bcs), b)
        snes.setJacobian(partial(J_nest, u, jacobian, preconditioner, bcs), A, P)
    else:
        raise ValueError(f"Unsupported SNES type {assembly_type}")
    return snes, x
