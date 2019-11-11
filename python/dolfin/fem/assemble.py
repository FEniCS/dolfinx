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
    return cpp.la.create_vector(_create_cpp_form(L).function_space(0).dofmap.index_map)


def create_vector_block(L: typing.List[typing.Union[Form, cpp.fem.Form]]) -> PETSc.Vec:
    maps = [form.function_space(0).dofmap.index_map for form in _create_cpp_form(L)]
    return cpp.fem.create_vector_block(maps)


def create_vector_nest(L: typing.List[typing.Union[Form, cpp.fem.Form]]) -> PETSc.Vec:
    maps = [form.function_space(0).dofmap.index_map for form in _create_cpp_form(L)]
    return cpp.fem.create_vector_nest(maps)


# -- Matrix instantiation ----------------------------------------------------


def create_matrix(a: typing.Union[Form, cpp.fem.Form]) -> PETSc.Mat:
    return cpp.fem.create_matrix(_create_cpp_form(a))


def create_matrix_block(a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]]) -> PETSc.Mat:
    return cpp.fem.create_matrix_block(_create_cpp_form(a))


def create_matrix_nest(a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]]) -> PETSc.Mat:
    return cpp.fem.create_matrix_nest(_create_cpp_form(a))


# -- Scalar assembly ---------------------------------------------------------


def assemble_scalar(M: typing.Union[Form, cpp.fem.Form]) -> PETSc.ScalarType:
    """Assemble functional. The returned value is local and not accumulated
    across processes.

    """
    return cpp.fem.assemble_scalar(_create_cpp_form(M))

# -- Vector assembly ---------------------------------------------------------


@functools.singledispatch
def assemble_vector(L: typing.Union[Form, cpp.fem.Form]) -> PETSc.Vec:
    """Assemble linear form into a new PETSc vector. The returned vector is
    not finalised, i.e. ghost values are not accumulated on the owning
    processes.

    """
    b = cpp.la.create_vector(_create_cpp_form(L).function_space(0).dofmap.index_map)
    with b.localForm() as b_local:
        b_local.set(0.0)
    cpp.fem.assemble_vector(b, _create_cpp_form(L))
    return b


@assemble_vector.register(PETSc.Vec)
def _(b: PETSc.Vec, L: typing.Union[Form, cpp.fem.Form]) -> PETSc.Vec:
    """Assemble linear form into an existing PETSc vector. The vector is not
    zeroed before assembly and it is not finalised, qi.e. ghost values are
    not accumulated on the owning processes.

    """
    cpp.fem.assemble_vector(b, _create_cpp_form(L))
    return b


@functools.singledispatch
def assemble_vector_nest(L: typing.Union[Form, cpp.fem.Form]) -> PETSc.Vec:
    """Assemble linear forms into a new nested PETSc (VecNest) vector. The
    returned vector is not finalised, i.e. ghost values are not accumulated
    on the owning processes.

    """
    maps = [form.function_space(0).dofmap.index_map for form in _create_cpp_form(L)]
    b = cpp.fem.create_vector_nest(maps)
    for b_sub in b.getNestSubVecs():
        with b_sub.localForm() as b_local:
            b_local.set(0.0)
    return assemble_vector_nest(b, L)


@assemble_vector_nest.register(PETSc.Vec)
def _(b: PETSc.Vec, L: typing.List[typing.Union[Form, cpp.fem.Form]]) -> PETSc.Vec:
    """Assemble linear forms into a nested PETSc (VecNest) vector. The vector is not
    zeroed before assembly and it is not finalised, qi.e. ghost values
    are not accumulated on the owning processes.

    """
    for b_sub, L_sub in zip(b.getNestSubVecs(), _create_cpp_form(L)):
        cpp.fem.assemble_vector(b_sub, L_sub)
    return b


# FIXME: Revise this interface
@functools.singledispatch
def assemble_vector_block(L: typing.List[typing.Union[Form, cpp.fem.Form]],
                          a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]],
                          bcs: typing.List[DirichletBC],
                          x0: typing.Optional[PETSc.Vec] = None,
                          scale: float = 1.0) -> PETSc.Vec:
    """Assemble linear forms into a monolithic vector. The vector is not
    zeroed and it is not finalised, i.e. ghost values are not
    accumulated.

    """
    maps = [form.function_space(0).dofmap.index_map for form in _create_cpp_form(L)]
    b = cpp.fem.create_vector_block(maps)
    b.set(0.0)
    return assemble_vector_block(b, L, a, bcs, x0, scale)


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
    maps = [form.function_space(0).dofmap.index_map for form in _create_cpp_form(L)]
    if x0 is not None:
        x0_local = cpp.la.get_local_vectors(x0, maps)
        x0_sub = x0_local
    else:
        x0_local = []
        x0_sub = [None] * len(maps)

    bcs1 = cpp.fem.bcs_cols(_create_cpp_form(a), bcs)
    b_local = cpp.la.get_local_vectors(b, maps)
    for b_sub, L_sub, a_sub, bc in zip(b_local, L, a, bcs1):
        cpp.fem.assemble_vector(b_sub, _create_cpp_form(L_sub))
        cpp.fem.apply_lifting(b_sub, _create_cpp_form(a_sub), bc, x0_local, scale)

    cpp.la.scatter_local_vectors(b, b_local, maps)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    bcs0 = cpp.fem.bcs_rows(_create_cpp_form(L), bcs)
    offset = 0
    b_array = b.getArray(readonly=False)
    for submap, bc, _x0 in zip(maps, bcs0, x0_sub):
        size = submap.size_local * submap.block_size
        cpp.fem.set_bc(b_array[offset:offset + size], bc, _x0, scale)
        offset += size

    return b


# -- Matrix assembly ---------------------------------------------------------


@functools.singledispatch
def assemble_matrix(a: typing.Union[Form, cpp.fem.Form],
                    bcs: typing.List[DirichletBC] = [],
                    diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    A = cpp.fem.create_matrix(_create_cpp_form(a))
    A.zeroEntries()
    return assemble_matrix(A, a, bcs, diagonal)


@assemble_matrix.register(PETSc.Mat)
def _(A: PETSc.Mat,
      a: typing.Union[Form, cpp.fem.Form],
      bcs: typing.List[DirichletBC] = [],
      diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    _a = _create_cpp_form(a)
    cpp.fem.assemble_matrix(A, _a, bcs)
    if _a.function_space(0) == _a.function_space(1):
        cpp.fem.add_diagonal(A, _a.function_space(0), bcs, diagonal)
    return A


# FIXME: Revise this interface
@functools.singledispatch
def assemble_matrix_nest(a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]],
                         bcs: typing.List[DirichletBC],
                         diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    A = cpp.fem.create_matrix_nest(_create_cpp_form(a))
    A.zeroEntries()
    assemble_matrix_nest(A, a, bcs, diagonal)
    return A


@assemble_matrix_nest.register(PETSc.Mat)
def _(A: PETSc.Mat,
      a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]],
      bcs: typing.List[DirichletBC],
      diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    _a = _create_cpp_form(a)
    for i, a_row in enumerate(_a):
        for j, a_block in enumerate(a_row):
            if a_block is not None:
                Asub = A.getNestSubMatrix(i, j)
                assemble_matrix(Asub, a_block, bcs)
    return A


# FIXME: Revise this interface
@functools.singledispatch
def assemble_matrix_block(a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]],
                          bcs: typing.List[DirichletBC],
                          diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    A = cpp.fem.create_matrix_block(_create_cpp_form(a))
    A.zeroEntries()
    return assemble_matrix_block(A, a, bcs, diagonal)


@assemble_matrix_block.register(PETSc.Mat)
def _(A: PETSc.Mat,
      a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]],
      bcs: typing.List[DirichletBC],
      diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    _a = _create_cpp_form(a)
    V = cpp.fem.block_function_spaces(_a)
    is_rows = cpp.la.create_petsc_index_sets([Vsub.dofmap.index_map for Vsub in V[0]])
    is_cols = cpp.la.create_petsc_index_sets([Vsub.dofmap.index_map for Vsub in V[1]])
    for i, a_row in enumerate(_a):
        for j, a_sub in enumerate(a_row):
            if a_sub is not None:
                Asub = A.getLocalSubMatrix(is_rows[i], is_cols[j])
                assemble_matrix(Asub, a_sub, bcs, diagonal)
                A.restoreLocalSubMatrix(is_rows[i], is_cols[j], Asub)
    return A


# -- Modifiers for Dirichlet conditions ---------------------------------------

def apply_lifting(b: PETSc.Vec,
                  a: typing.List[typing.Union[Form, cpp.fem.Form]],
                  bcs: typing.List[typing.List[DirichletBC]],
                  x0: typing.Optional[typing.List[PETSc.Vec]] = [],
                  scale: float = 1.0) -> None:
    """Modify RHS vector b for lifting of Dirichlet boundary conditions.
    It modifies b such that:

        b <- b - scale * A_j (g_j - x0_j)

    where j is a block (nest) index. For a non-blocked problem j = 0.
    The boundary conditions bcs are on the trial spaces V_j. The forms
    in [a] must have the same test space as L (from which b was built),
    but the trial space may differ. If x0 is not supplied, then it is
    treated as zero.

    Ghost contributions are not accumulated (not sent to owner). Caller
    is responsible for calling VecGhostUpdateBegin/End.
    """
    cpp.fem.apply_lifting(b, _create_cpp_form(a), bcs, x0, scale)


def apply_lifting_nest(b: PETSc.Vec,
                       a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]],
                       bcs: typing.List[DirichletBC],
                       x0: typing.Optional[PETSc.Vec] = None,
                       scale: float = 1.0) -> PETSc.Vec:
    """Modify nested vector for lifting of Dirichlet boundary conditions.

    """
    if x0 is not None:
        _x0 = x0.getNestSubVecs()
    else:
        _x0 = []
    _a = _create_cpp_form(a)
    bcs1 = cpp.fem.bcs_cols(_a, bcs)
    for b_sub, a_sub, bc1 in zip(b.getNestSubVecs(), _a, bcs1):
        cpp.fem.apply_lifting(b_sub, a_sub, bc1, _x0, scale)

    return b


def set_bc(b: PETSc.Vec,
           bcs: typing.List[DirichletBC],
           x0: typing.Optional[PETSc.Vec] = None,
           scale: float = 1.0) -> None:
    """Insert boundary condition values into vector. Only local (owned)
    entries are set, hence communication after calling this function is
    not required unless ghost entries need to be updated to the boundary
    condition value.

    """
    cpp.fem.set_bc(b, bcs, x0, scale)


def set_bc_nest(b: PETSc.Vec,
                bcs: typing.List[typing.List[DirichletBC]],
                x0: typing.Optional[PETSc.Vec] = None,
                scale: float = 1.0) -> None:
    """Insert boundary condition values into nested vector. Only local (owned)
    entries are set, hence communication after calling this function is
    not required unless the ghost entries need to be updated to the
    boundary condition value.

    """
    _b = b.getNestSubVecs()
    if x0 is not None:
        _x0 = x0.getNestSubVecs()
    else:
        _x0 = [None] * len(_b)
    for b_sub, bc, x_sub in zip(_b, bcs, _x0):
        cpp.fem.set_bc(b_sub, bc, x_sub, scale)
