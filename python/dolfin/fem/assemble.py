# Copyright (C) 2018-2019 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Assembly functions for variational forms."""

import functools
import typing

from petsc4py import PETSc

import ufl

from dolfin import cpp
from dolfin.fem.dirichletbc import DirichletBC
from dolfin.fem.form import Form


def _create_cpp_form(form):
    """Recursively look for ufl.Forms and convert to dolfin.fem.Form, otherwise
    return form argument
    """
    if isinstance(form, ufl.Form):
        return Form(form)._cpp_object
    elif isinstance(form, (tuple, list)):
        return list(map(lambda sub_form: _create_cpp_form(sub_form), form))
    return form


# -- Vector instantiation ----------------------------------------------------


def create_vector(L: typing.Union[Form, cpp.fem.Form]) -> PETSc.Vec:
    b = cpp.la.create_vector(_create_cpp_form(L).function_space(0).dofmap.index_map)
    return b


def create_vector_block(L: typing.List[typing.Union[Form, cpp.fem.Form]]) -> PETSc.Vec:
    b = cpp.fem.create_vector(_create_cpp_form(L))
    return b


def create_vector_nest(L: typing.List[typing.Union[Form, cpp.fem.Form]]) -> PETSc.Vec:
    b = cpp.fem.create_vector_nest(_create_cpp_form(L))
    return b


# -- Matrix instantiation ----------------------------------------------------


def create_matrix(a: typing.Union[Form, cpp.fem.Form]) -> PETSc.Mat:
    A = cpp.fem.create_matrix(_create_cpp_form(a))
    return A


def create_matrix_block(a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]]) -> PETSc.Mat:
    A = cpp.fem.create_matrix_block(_create_cpp_form(a))
    return A


def create_matrix_nest(a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]]) -> PETSc.Mat:
    A = cpp.fem.create_matrix_nest(_create_cpp_form(a))
    return A


# -- Scalar assembly ---------------------------------------------------------


def assemble_scalar(M: typing.Union[Form, cpp.fem.Form]) -> PETSc.ScalarType:
    """Assemble functional. The returned value is local and not accumulated
    across processes.

    """
    return cpp.fem.assemble_scalar(_create_cpp_form(M))

# -- Vector assembly ---------------------------------------------------------


@functools.singledispatch
def assemble_vector(L: typing.Union[Form, cpp.fem.Form]) -> PETSc.Vec:
    """Assemble linear form into a vector. The returned vector is not
    finalised, i.e. ghost values are not accumulated.

    """
    b = cpp.la.create_vector(_create_cpp_form(L).function_space(0).dofmap.index_map)
    with b.localForm() as b_local:
        b_local.set(0.0)
    cpp.fem.assemble_vector(b, _create_cpp_form(L))
    return b


@assemble_vector.register(PETSc.Vec)
def _(b: PETSc.Vec, L: typing.Union[Form, cpp.fem.Form]) -> PETSc.Vec:
    """Re-assemble linear form into a vector.

    The vector is not zeroed and it is not finalised, i.e. ghost values
    are not accumulated.

    """
    cpp.fem.assemble_vector(b, _create_cpp_form(L))
    return b


# FIXME: Revise this interface
@functools.singledispatch
def assemble_vector_nest(L: typing.List[typing.Union[Form, cpp.fem.Form]],
                         a,
                         bcs: typing.List[DirichletBC],
                         x0: typing.Optional[PETSc.Vec] = None,
                         scale: float = 1.0) -> PETSc.Vec:
    """Assemble linear forms into a nested (VecNest) vector. The vector is
    not zeroed and it is not finalised, i.e. ghost values are not
    accumulated.

    """
    b = cpp.fem.create_vector_nest(_create_cpp_form(L))
    return assemble_vector_nest(b, L, a, bcs, x0, scale)

    # cpp.fem.assemble_vector(b, _create_cpp_form(L), _create_cpp_form(a), bcs, x0, scale)
    # if x0 is not None:
    #     _x0 = x0.getNestSubVecs()
    # for i in range(len(L)):
    #     _b = b.getNestSubVec(i)
    #     cpp.fem.assemble_vector(_b, _create_cpp_form(L[i]))
    #     cpp.fem.apply_lifting(_b, _create_cpp_form(a[i]), bcs1,
    #                           _x0, scale)
    #     cpp.fem.set_bc(_b, bcs[i], _x0[i])
    # return b


@assemble_vector_nest.register(PETSc.Vec)
def _(b: PETSc.Vec,
      L: typing.List[typing.Union[Form, cpp.fem.Form]],
      a,
      bcs: typing.List[DirichletBC],
      x0: typing.Optional[PETSc.Vec] = None,
      scale: float = 1.0) -> PETSc.Vec:
    """Assemble linear forms into a nested (VecNest) vector. The vector is
    not zeroed and it is not finalised, i.e. ghost values are not
    accumulated.

    """
    cpp.fem.assemble_vector(b, _create_cpp_form(L), _create_cpp_form(a), bcs, x0, scale)
    return b

    if x0 is not None:
        _x0 = x0.getNestSubVecs()
    else:
        _x0 = [None]*len(L)
    _b = b.getNestSubVecs()
    bcs0 = cpp.fem.bcs_rows(_create_cpp_form(L), bcs)
    bcs1 = cpp.fem.bcs_cols(_create_cpp_form(a), bcs)

    L = _create_cpp_form(L)
    a = _create_cpp_form(a)

    for i in range(len(L)):
        cpp.fem.assemble_vector(_b[i], L[i])
        if x0 is not None:
            cpp.fem.apply_lifting(_b[i], a[i], bcs1[i], _x0, scale)
        else:
            cpp.fem.apply_lifting(_b[i], a[i], bcs1[i], [], scale)

        _b[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        if a[i][0] == []:
            # FIXME: this is a hack to handle the case that no bilinear forms
            # have been supplied, which may happen in a Newton iteration.
            # Needs to be fixed for nested systems
            cpp.fem.set_bc(_b[i], bcs0[0], _x0[i], scale)
        else:
            for j in range(len(a[i])):
                if a[i][j] is not None:
                    if L[i].function_space(0) == a[i][j].function_space(1):
                        cpp.fem.set_bc(_b[i], bcs0[i], _x0[i], scale)

        # for (std::size_t j = 0; j < a[i].size(); ++j)
        # {
        #   if (a[i][j])
        #   {
        #     if (*L[i]->function_space(0) == *a[i][j]->function_space(1))
        #       set_bc(b_sub, bcs0[i], x0_sub[i], scale);
        #   }
        # }

    return b


@functools.singledispatch
def assemble_vector_block(L: typing.List[typing.Union[Form, cpp.fem.Form]],
                          a,
                          bcs: typing.List[DirichletBC],
                          x0: typing.Optional[PETSc.Vec] = None,
                          scale: float = 1.0) -> PETSc.Vec:
    """Assemble linear forms into a monolithic vector. The vector is not
    zeroed and it is not finalised, i.e. ghost values are not
    accumulated.

    """
    b = cpp.fem.create_vector(_create_cpp_form(L))
    cpp.fem.assemble_vector(b, _create_cpp_form(L), _create_cpp_form(a), bcs, x0, scale)
    return b


@assemble_vector_block.register(PETSc.Vec)
def _(b: PETSc.Vec,
      L: typing.List[typing.Union[Form, cpp.fem.Form]],
      a,
      bcs: typing.List[DirichletBC],
      x0: typing.Optional[PETSc.Vec] = None,
      scale: float = 1.0) -> PETSc.Vec:
    """Assemble linear forms into a monolithic vector. The vector is not
    zeroed and it is not finalised, i.e. ghost values are not
    accumulated.

    """
    cpp.fem.assemble_vector(b, _create_cpp_form(L), _create_cpp_form(a), bcs, x0, scale)
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
    cpp.fem.reassemble_vector(b, _create_cpp_form(L), _create_cpp_form(a), bcs, x0, scale)
    return b


# -- Matrix assembly ---------------------------------------------------------


@functools.singledispatch
def assemble_matrix(a,
                    bcs: typing.List[DirichletBC] = [],
                    diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    A = cpp.fem.create_matrix(_create_cpp_form(a))
    A.zeroEntries()
    cpp.fem.assemble_matrix(A, _create_cpp_form(a), bcs, diagonal)
    return A


@assemble_matrix.register(PETSc.Mat)
def _(A, a, bcs: typing.List[DirichletBC] = [],
      diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    cpp.fem.assemble_matrix(A, _create_cpp_form(a), bcs, diagonal)
    return A


# FIXME: Revise this interface
@functools.singledispatch
def assemble_matrix_nest(a,
                         bcs: typing.List[DirichletBC],
                         diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    A = cpp.fem.create_matrix_nest(_create_cpp_form(a))
    A.zeroEntries()
    cpp.fem.assemble_blocked_matrix(A, _create_cpp_form(a), bcs, diagonal)
    return A


@assemble_matrix_nest.register(PETSc.Mat)
def _(A: PETSc.Mat,
      a,
      bcs: typing.List[DirichletBC],
      diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    cpp.fem.assemble_blocked_matrix(A, _create_cpp_form(a), bcs, diagonal)
    return A


# FIXME: Revise this interface
@functools.singledispatch
def assemble_matrix_block(a,
                          bcs: typing.List[DirichletBC],
                          diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    A = cpp.fem.create_matrix_block(_create_cpp_form(a))
    A.zeroEntries()
    cpp.fem.assemble_blocked_matrix(A, _create_cpp_form(a), bcs, diagonal)
    return A


@assemble_matrix_block.register(PETSc.Mat)
def _(A: PETSc.Mat,
      a,
      bcs: typing.List[DirichletBC],
      diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    cpp.fem.assemble_blocked_matrix(A, _create_cpp_form(a), bcs, diagonal)
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
    cpp.fem.apply_lifting(b, _create_cpp_form(a), bcs, x0, scale)


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
