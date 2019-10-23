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


# FIXME: This really shouldn't be needed
def convert_ufl_forms_to_dolfin_forms(f):
    """Function decorator to wrap all ufl.Form arguments to dolfin.fem.Form.
    This decorator prevents having to do this conversion in each of the members
    of this module.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        args = _create_cpp_form(args)
        return f(*args, **kwargs)
    return wrapper


# -- Vector instantiation ----------------------------------------------------


@convert_ufl_forms_to_dolfin_forms
def create_vector(L: typing.Union[Form, cpp.fem.Form]) -> PETSc.Vec:
    b = cpp.la.create_vector(L.function_space(0).dofmap.index_map)
    return b


@convert_ufl_forms_to_dolfin_forms
def create_vector_block(L: typing.List[typing.Union[Form, cpp.fem.Form]]) -> PETSc.Vec:
    b = cpp.fem.create_vector(L)
    return b


@convert_ufl_forms_to_dolfin_forms
def create_vector_nest(L: typing.List[typing.Union[Form, cpp.fem.Form]]) -> PETSc.Vec:
    b = cpp.fem.create_vector_nest(L)
    return b


# -- Matrix instantiation ----------------------------------------------------


@convert_ufl_forms_to_dolfin_forms
def create_matrix(a: typing.Union[Form, cpp.fem.Form]) -> PETSc.Mat:
    A = cpp.fem.create_matrix(a)
    return A


@convert_ufl_forms_to_dolfin_forms
def create_matrix_block(a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]]) -> PETSc.Mat:
    A = cpp.fem.create_matrix_block(a)
    return A


@convert_ufl_forms_to_dolfin_forms
def create_matrix_nest(a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]]) -> PETSc.Mat:
    A = cpp.fem.create_matrix_nest(a)
    return A


# -- Scalar assembly ---------------------------------------------------------


@convert_ufl_forms_to_dolfin_forms
def assemble_scalar(M: typing.Union[Form, cpp.fem.Form]) -> PETSc.ScalarType:
    """Assemble functional. The returned value is local and not accumulated
    across processes.

    """
    return cpp.fem.assemble_scalar(M)

# -- Vector assembly ---------------------------------------------------------


@convert_ufl_forms_to_dolfin_forms
@functools.singledispatch
def assemble_vector(L: typing.Union[Form, cpp.fem.Form]) -> PETSc.Vec:
    """Assemble linear form into a vector. The returned vector is not
    finalised, i.e. ghost values are not accumulated.

    """
    b = cpp.la.create_vector(L.function_space(0).dofmap.index_map)
    with b.localForm() as b_local:
        b_local.set(0.0)
    cpp.fem.assemble_vector(b, L)
    return b


@convert_ufl_forms_to_dolfin_forms
@assemble_vector.register(PETSc.Vec)
def _(b: PETSc.Vec, L: typing.Union[Form, cpp.fem.Form]) -> PETSc.Vec:
    """Re-assemble linear form into a vector.

    The vector is not zeroed and it is not finalised, i.e. ghost values
    are not accumulated.

    """
    cpp.fem.assemble_vector(b, L)
    return b


# FIXME: Revise this interface
@convert_ufl_forms_to_dolfin_forms
def assemble_vector_nest(L: typing.List[typing.Union[Form, cpp.fem.Form]],
                         a,
                         bcs: typing.List[DirichletBC],
                         x0: typing.Optional[PETSc.Vec] = None,
                         scale: float = 1.0) -> PETSc.Vec:
    """Assemble linear forms into a nested (VecNest) vector. The vector is
    not zeroed and it is not finalised, i.e. ghost values are not
    accumulated.

    """
    b = cpp.fem.create_vector_nest(L)
    cpp.fem.assemble_vector(b, L, a, bcs, x0, scale)
    return b


# FIXME: Revise this interface
@convert_ufl_forms_to_dolfin_forms
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
    b = cpp.fem.create_vector(L)
    cpp.fem.assemble_vector(b, L, a, bcs, x0, scale)
    return b


@convert_ufl_forms_to_dolfin_forms
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
    cpp.fem.assemble_vector(b, L, a, bcs, x0, scale)
    return b


# FIXME: Revise this interface
@convert_ufl_forms_to_dolfin_forms
def reassemble_vector(b: PETSc.Vec,
                      L,
                      a=[],
                      bcs: typing.List[DirichletBC] = [],
                      x0: typing.Optional[PETSc.Vec] = None,
                      scale: float = 1.0) -> PETSc.Vec:
    """Re-assemble linear forms into a block vector, with modification for Dirichlet
    boundary conditions

    """
    cpp.fem.reassemble_vector(b, L, a, bcs, x0, scale)
    return b


# -- Matrix assembly ---------------------------------------------------------


@convert_ufl_forms_to_dolfin_forms
@functools.singledispatch
def assemble_matrix(a,
                    bcs: typing.List[DirichletBC] = [],
                    diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    A = cpp.fem.create_matrix(a)
    A.zeroEntries()
    cpp.fem.assemble_matrix(A, a, bcs, diagonal)
    return A


@convert_ufl_forms_to_dolfin_forms
@assemble_matrix.register(PETSc.Mat)
def _(A, a, bcs: typing.List[DirichletBC] = [],
      diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    cpp.fem.assemble_matrix(A, a, bcs, diagonal)
    return A


# FIXME: Revise this interface
@convert_ufl_forms_to_dolfin_forms
def assemble_matrix_nest(a,
                         bcs: typing.List[DirichletBC],
                         diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    A = cpp.fem.create_matrix_nest(a)
    A.zeroEntries()
    cpp.fem.assemble_blocked_matrix(A, a, bcs, diagonal)
    return A


# FIXME: Revise this interface
@convert_ufl_forms_to_dolfin_forms
@functools.singledispatch
def assemble_matrix_block(a,
                          bcs: typing.List[DirichletBC],
                          diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    A = cpp.fem.create_matrix_block(a)
    A.zeroEntries()
    cpp.fem.assemble_blocked_matrix(A, a, bcs, diagonal)
    return A


@convert_ufl_forms_to_dolfin_forms
@assemble_matrix_block.register(PETSc.Mat)
def _(A: PETSc.Mat,
      a,
      bcs: typing.List[DirichletBC],
      diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    cpp.fem.assemble_blocked_matrix(A, a, bcs, diagonal)
    return A


# -- Modifiers for Dirichlet conditions ---------------------------------------

# FIXME: Explain in docstring order of calling this function and
# parallel udpdating w.r.t assembly of L into b.
@convert_ufl_forms_to_dolfin_forms
def apply_lifting(b: PETSc.Vec,
                  a: typing.List[typing.Union[Form, cpp.fem.Form]],
                  bcs: typing.List[typing.List[DirichletBC]],
                  x0: typing.Optional[typing.List[PETSc.Vec]] = [],
                  scale: float = 1.0) -> None:
    """Modify vector for lifting of Dirichlet boundary conditions.

    """
    cpp.fem.apply_lifting(b, a, bcs, x0, scale)


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
