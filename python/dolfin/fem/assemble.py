# -*- coding: utf-8 -*-
# Copyright (C) 2018-2019 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Assembly functions for variational forms."""

import functools
import typing

from petsc4py import PETSc

from dolfin import cpp
from dolfin.fem.assembling import _create_cpp_form
from dolfin.fem.dirichletbc import DirichletBC
from dolfin.fem.form import Form


@functools.singledispatch
def assemble(M: typing.Union[Form, cpp.fem.Form]
             ) -> typing.Union[float, PETSc.Mat, PETSc.Vec]:
    """Assemble a form over mesh"""
    _M = _create_cpp_form(M)
    if _M.rank() == 0:
        return cpp.fem.assemble_scalar(_M)
    elif _M.rank() == 1:
        b = cpp.la.create_vector(_M.function_space(0).dofmap().index_map())
        assemble(b, _M)
        return b
    elif _M.rank() == 2:
        A = cpp.fem.create_matrix(_M)
        assemble(A, _M)
        return A
    else:
        raise RuntimeError("Form rank not supported by assembler.")


@assemble.register(PETSc.Vec)
def _assemble_vector(b: PETSc.Vec,
                     L: typing.Union[Form, cpp.fem.Form]) -> PETSc.Vec:
    """Re-assemble linear form into a vector."""
    L_cpp = _create_cpp_form(L)
    cpp.fem.assemble_vector(b, L_cpp)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return b


@assemble.register(PETSc.Mat)
def _assemble_matrix(A: PETSc.Mat,
                     a: typing.Union[Form, cpp.fem.Form],
                     bcs=[],
                     diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear form into a vector, with rows and columns with Dirichlet
    boundary conditions zeroed.

    """
    A.zeroEntries()
    a_cpp = _create_cpp_form(a)
    cpp.fem.assemble_matrix(A, a_cpp, bcs, diagonal)
    A.assemble()
    return A


@functools.singledispatch
def assemble_vector(L: typing.Union[Form, cpp.fem.Form]) -> PETSc.Vec:
    """Assemble linear form into a vector."""
    L_cpp = _create_cpp_form(L)
    b = cpp.la.create_vector(L_cpp.function_space(0).dofmap().index_map())
    cpp.fem.assemble_vector(b, L_cpp)
    return b


@assemble_vector.register(PETSc.Vec)
def _reassemble_vector(b: PETSc.Vec,
                       L: typing.Union[Form, cpp.fem.Form]) -> PETSc.Vec:
    """Re-assemble linear form into a vector."""
    L_cpp = _create_cpp_form(L)
    cpp.fem.assemble_vector(b, L_cpp)
    return b


def assemble_vector_nest(
        L: typing.List[typing.Union[Form, cpp.fem.Form]],
        a,
        bcs: typing.List[DirichletBC],
        x0: typing.Optional[PETSc.Vec] = None,
        scale: float = 1.0,
) -> PETSc.Vec:
    """Assemble linear forms into a nested vector"""
    L_cpp = [_create_cpp_form(form) for form in L]
    a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    b = cpp.fem.create_vector_nest(L_cpp)
    cpp.fem.assemble_vector(b, L_cpp, a_cpp, bcs, x0, scale)
    return b


def assemble_vector_block(
        L: typing.List[typing.Union[Form, cpp.fem.Form]],
        a,
        bcs: typing.List[DirichletBC],
        x0: typing.Optional[PETSc.Vec] = None,
        scale: float = 1.0,
) -> PETSc.Vec:
    """Assemble linear forms into a monolithic vector"""
    L_cpp = [_create_cpp_form(form) for form in L]
    a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    b = cpp.fem.create_vector(L_cpp)
    cpp.fem.assemble_vector(b, L_cpp, a_cpp, bcs, x0, scale)
    return b


def reassemble_vector(
        b: PETSc.Vec,
        L,
        a=[],
        bcs: typing.List[DirichletBC] = [],
        x0: typing.Optional[PETSc.Vec] = None,
        scale: float = 1.0,
) -> PETSc.Vec:
    """Re-assemble linear forms into a block vector, with modification for Dirichlet
    boundary conditions

    """
    L_cpp = [_create_cpp_form(form) for form in L]
    a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    cpp.fem.reassemble_vector(b, L_cpp, a_cpp, bcs, x0, scale)
    return b


def assemble_matrix_nest(a,
                         bcs: typing.List[DirichletBC],
                         diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    A = cpp.fem.create_matrix_nest(a_cpp)
    A.zeroEntries()
    cpp.fem.assemble_blocked_matrix(A, a_cpp, bcs, diagonal)
    return A


def assemble_matrix_block(a,
                          bcs: typing.List[DirichletBC],
                          diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    A = cpp.fem.create_matrix_block(a_cpp)
    A.zeroEntries()
    cpp.fem.assemble_blocked_matrix(A, a_cpp, bcs, diagonal)
    return A


def assemble_matrix(a, bcs: typing.List[DirichletBC],
                    diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear form into matrix."""
    a_cpp = _create_cpp_form(a)
    A = cpp.fem.create_matrix(a_cpp)
    assemble(A, a_cpp, bcs, diagonal)
    return A


def apply_lifting(
        b: PETSc.Vec,
        a: typing.List[typing.Union[Form, cpp.fem.Form]],
        bcs: typing.List[typing.List[DirichletBC]],
        x0: typing.Optional[typing.List[PETSc.Vec]] = [],
        scale: float = 1.0) -> None:
    """Modify vector for lifting of boundary conditions."""
    a_cpp = [_create_cpp_form(form) for form in a]
    cpp.fem.apply_lifting(b, a_cpp, bcs, x0, scale)


def set_bc(b: PETSc.Vec, bcs: typing.List[DirichletBC],
           x0: typing.Optional[PETSc.Vec] = None,
           scale: float = 1.0) -> None:
    """Insert boundary condition values into vector"""
    cpp.fem.set_bc(b, bcs, x0, scale)
