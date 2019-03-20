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
from dolfin.fem.dirichletbc import DirichletBC
from dolfin.fem.form import Form


def _create_cpp_form(form: typing.Union[Form, cpp.fem.Form]) -> cpp.fem.Form:
    """Create a compiled Form from a UFL form"""
    if form is None:
        return None
    elif isinstance(form, cpp.fem.Form):
        return form
    else:
        # FIXME: Attach cpp Form to UFL Form to avoid re-processing
        form = Form(form)
        return form._cpp_object


# -- Scalar assembly ---------------------------------------------------------

def assemble_scalar(M: typing.Union[Form, cpp.fem.Form]) -> PETSc.ScalarType:
    """Assemble functional. The returned value is local and not accumulated
    across processes.

    """
    M_cpp = _create_cpp_form(M)
    return cpp.fem.assemble_scalar(M_cpp)

# -- Vector assembly ---------------------------------------------------------


@functools.singledispatch
def assemble_vector(L: typing.Union[Form, cpp.fem.Form]) -> PETSc.Vec:
    """Assemble linear form into a vector. The returned vector is not
    finalised, i.e. ghost values are not accumulated.

    """
    L_cpp = _create_cpp_form(L)
    b = cpp.la.create_vector(L_cpp.function_space(0).dofmap().index_map())
    with b.localForm() as b_local:
        b_local.set(0.0)
    cpp.fem.assemble_vector(b, L_cpp)
    return b


@assemble_vector.register(PETSc.Vec)
def _(b: PETSc.Vec, L: typing.Union[Form, cpp.fem.Form]) -> PETSc.Vec:
    """Re-assemble linear form into a vector.

    The vector is not zeroed and it is not finalised, i.e. ghost values
    are not accumulated.

    """
    L_cpp = _create_cpp_form(L)
    cpp.fem.assemble_vector(b, L_cpp)
    return b


# FIXME: Revise this interface
def assemble_vector_nest(L: typing.List[typing.Union[Form, cpp.fem.Form]],
                         a,
                         bcs: typing.List[DirichletBC],
                         x0: typing.Optional[PETSc.Vec] = None,
                         scale: float = 1.0) -> PETSc.Vec:
    """Assemble linear forms into a nested (VecNest) vector. The vector is
    not zeroed and it is not finalised, i.e. ghost values are not
    accumulated.

    """
    L_cpp = [_create_cpp_form(form) for form in L]
    a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    b = cpp.fem.create_vector_nest(L_cpp)
    cpp.fem.assemble_vector(b, L_cpp, a_cpp, bcs, x0, scale)
    return b


# FIXME: Revise this interface
def assemble_vector_block(L: typing.List[typing.Union[Form, cpp.fem.Form]],
                          a,
                          bcs: typing.List[DirichletBC],
                          x0: typing.Optional[PETSc.Vec] = None,
                          scale: float = 1.0) -> PETSc.Vec:
    """Assemble linear forms into a monolithic vector. The vector is not
    zeroed and it is not finalised, i.e. ghost values are not
    accumulated.

    """
    L_cpp = [_create_cpp_form(form) for form in L]
    a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    b = cpp.fem.create_vector(L_cpp)
    cpp.fem.assemble_vector(b, L_cpp, a_cpp, bcs, x0, scale)
    return b


# FIXME: Revise this interface
def reassemble_vector(b: PETSc.Vec,
                      L,
                      a=[],
                      bcs: typing.List[DirichletBC] = [],
                      x0: typing.Optional[PETSc.Vec] = None,
                      scale: float = 1.0) -> PETSc.Vec:
    """Re-assemble linear forms into a block vector, with modification for Dirichlet
    boundary conditions

    """
    L_cpp = [_create_cpp_form(form) for form in L]
    a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    cpp.fem.reassemble_vector(b, L_cpp, a_cpp, bcs, x0, scale)
    return b


# -- Matrix assembly ---------------------------------------------------------


@functools.singledispatch
def assemble_matrix(a,
                    bcs: typing.List[DirichletBC] = [],
                    diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    a_cpp = _create_cpp_form(a)
    A = cpp.fem.create_matrix(a_cpp)
    A.zeroEntries()
    cpp.fem.assemble_matrix(A, a_cpp, bcs, diagonal)
    return A


@assemble_matrix.register(PETSc.Mat)
def _(A, a, bcs: typing.List[DirichletBC] = [],
      diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    a_cpp = _create_cpp_form(a)
    cpp.fem.assemble_matrix(A, a_cpp, bcs, diagonal)
    return A


# FIXME: Revise this interface
def assemble_matrix_nest(a,
                         bcs: typing.List[DirichletBC],
                         diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    A = cpp.fem.create_matrix_nest(a_cpp)
    A.zeroEntries()
    cpp.fem.assemble_blocked_matrix(A, a_cpp, bcs, diagonal)
    return A


# FIXME: Revise this interface
def assemble_matrix_block(a,
                          bcs: typing.List[DirichletBC],
                          diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    A = cpp.fem.create_matrix_block(a_cpp)
    A.zeroEntries()
    cpp.fem.assemble_blocked_matrix(A, a_cpp, bcs, diagonal)
    return A


# -- Modifiers for Dirichlet conditions ---------------------------------------

# FIXME: Explain in docstring order of calling this function and
# parallel udpdating w.r.t assembly of L into b.
def apply_lifting(b: PETSc.Vec,
                  a: typing.List[typing.Union[Form, cpp.fem.Form]],
                  bcs: typing.List[typing.List[DirichletBC]],
                  x0: typing.Optional[typing.List[PETSc.Vec]] = [],
                  scale: float = 1.0) -> None:
    """Modify vector for lifting of Dirichlet boundary conditions.

    """
    a_cpp = [_create_cpp_form(form) for form in a]
    cpp.fem.apply_lifting(b, a_cpp, bcs, x0, scale)


def set_bc(b: PETSc.Vec,
           bcs: typing.List[DirichletBC],
           x0: typing.Optional[PETSc.Vec] = None,
           scale: float = 1.0) -> None:
    """Insert boundary condition values into vector. Only local (owned)
    entries are set, hence communication after calling this function is
    not required the ghost entries need to be updated to the boundary
    condition value.

    """
    cpp.fem.set_bc(b, bcs, x0, scale)
