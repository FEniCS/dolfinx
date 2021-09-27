# Copyright (C) 2018-2019 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Assembly functions for variational forms."""

import collections
import contextlib
import functools
import typing

import ufl
from dolfinx import cpp
from dolfinx.fem.dirichletbc import DirichletBC
from dolfinx.fem.form import Form
from petsc4py import PETSc


def _create_cpp_form(form):
    """Recursively look for ufl.Forms and convert to dolfinx.fem.Form, otherwise
    return form argument
    """
    if isinstance(form, Form):
        return form._cpp_object
    elif isinstance(form, ufl.Form):
        return Form(form)._cpp_object
    elif isinstance(form, (tuple, list)):
        return list(map(lambda sub_form: _create_cpp_form(sub_form), form))
    return form


# -- Packing constants and coefficients --------------------------------------

form_type = typing.Union[Form, cpp.fem.Form, ufl.Form,
                         collections.abc.Sequence[Form],
                         collections.abc.Sequence[cpp.fem.Form],
                         collections.abc.Sequence[ufl.Form]]


def pack_constants(form: form_type):
    """Compute form constants. If form is an array of forms, this
    function returns an array of form constants with the same shape as
    form.

    """
    def _pack(form):
        if form is None:
            return None
        elif isinstance(form, (tuple, list)):
            return list(map(lambda sub_form: _pack(sub_form), form))
        else:
            return cpp.fem.pack_constants(form)
    return _pack(_create_cpp_form(form))


def pack_coefficients(form: form_type):
    """Compute form coefficients. If form is an array of forms, this
    function returns an array of form coefficients with the same shape
    as form.

    """
    def _pack(form):
        if form is None:
            return None
        elif isinstance(form, (tuple, list)):
            return list(map(lambda sub_form: _pack(sub_form), form))
        else:
            return cpp.fem.pack_coefficients(form)
    return _pack(_create_cpp_form(form))


# -- Vector instantiation ----------------------------------------------------

def create_vector(L: typing.Union[Form, cpp.fem.Form]) -> PETSc.Vec:
    dofmap = _create_cpp_form(L).function_spaces[0].dofmap
    return cpp.la.create_vector(dofmap.index_map, dofmap.index_map_bs)


def create_vector_block(L: typing.List[typing.Union[Form, cpp.fem.Form]]) -> PETSc.Vec:
    maps = [(form.function_spaces[0].dofmap.index_map, form.function_spaces[0].dofmap.index_map_bs)
            for form in _create_cpp_form(L)]
    return cpp.fem.create_vector_block(maps)


def create_vector_nest(L: typing.List[typing.Union[Form, cpp.fem.Form]]) -> PETSc.Vec:
    maps = [(form.function_spaces[0].dofmap.index_map, form.function_spaces[0].dofmap.index_map_bs)
            for form in _create_cpp_form(L)]
    return cpp.fem.create_vector_nest(maps)


# -- Matrix instantiation ----------------------------------------------------

def create_matrix(a: typing.Union[Form, cpp.fem.Form], mat_type=None) -> PETSc.Mat:
    if mat_type is not None:
        return cpp.fem.create_matrix(_create_cpp_form(a), mat_type)
    else:
        return cpp.fem.create_matrix(_create_cpp_form(a))


def create_matrix_block(a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]]) -> PETSc.Mat:
    return cpp.fem.create_matrix_block(_create_cpp_form(a))


def create_matrix_nest(a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]]) -> PETSc.Mat:
    return cpp.fem.create_matrix_nest(_create_cpp_form(a))


Coefficients = collections.namedtuple('Coefficients', ['constants', 'coeffs'])

# -- Scalar assembly ---------------------------------------------------------


def assemble_scalar(M: typing.Union[Form, cpp.fem.Form],
                    coeffs=Coefficients(None, None)) -> PETSc.ScalarType:
    """Assemble functional. The returned value is local and not
    accumulated across processes.

    """
    _M = _create_cpp_form(M)
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(_M),
         coeffs[1] if coeffs[1] is not None else pack_coefficients(_M))
    return cpp.fem.assemble_scalar(_M, c[0], c[1])


# -- Vector assembly ---------------------------------------------------------

@functools.singledispatch
def assemble_vector(L: typing.Union[Form, cpp.fem.Form],
                    coeffs=Coefficients(None, None)) -> PETSc.Vec:
    """Assemble linear form into a new PETSc vector. The returned vector
    is not finalised, i.e. ghost values are not accumulated on the
    owning processes.

    """
    _L = _create_cpp_form(L)
    b = cpp.la.create_vector(_L.function_spaces[0].dofmap.index_map,
                             _L.function_spaces[0].dofmap.index_map_bs)
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(_L),
         coeffs[1] if coeffs[1] is not None else pack_coefficients(_L))
    with b.localForm() as b_local:
        b_local.set(0.0)
        cpp.fem.assemble_vector(b_local.array_w, _L, c[0], c[1])
    return b


@assemble_vector.register(PETSc.Vec)
def _(b: PETSc.Vec, L: typing.Union[Form, cpp.fem.Form], coeffs=Coefficients(None, None)) -> PETSc.Vec:
    """Assemble linear form into an existing PETSc vector. The vector is
    not zeroed before assembly and it is not finalised, i.e. ghost
    values are not accumulated on the owning processes.

    """
    _L = _create_cpp_form(L)
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(_L),
         coeffs[1] if coeffs[1] is not None else pack_coefficients(_L))
    with b.localForm() as b_local:
        cpp.fem.assemble_vector(b_local.array_w, _L, c[0], c[1])
    return b


@functools.singledispatch
def assemble_vector_nest(L: typing.Union[Form, cpp.fem.Form], coeffs=Coefficients(None, None)) -> PETSc.Vec:
    """Assemble linear forms into a new nested PETSc (VecNest) vector.
    The returned vector is not finalised, i.e. ghost values are not
    accumulated on the owning processes.

    """
    maps = [(form.function_spaces[0].dofmap.index_map, form.function_spaces[0].dofmap.index_map_bs)
            for form in _create_cpp_form(L)]
    b = cpp.fem.create_vector_nest(maps)
    for b_sub in b.getNestSubVecs():
        with b_sub.localForm() as b_local:
            b_local.set(0.0)
    return assemble_vector_nest(b, L, coeffs)


@assemble_vector_nest.register(PETSc.Vec)
def _(b: PETSc.Vec, L: typing.List[typing.Union[Form, cpp.fem.Form]], coeffs=Coefficients(None, None)) -> PETSc.Vec:
    """Assemble linear forms into a nested PETSc (VecNest) vector. The
    vector is not zeroed before assembly and it is not finalised, i.e.
    ghost values are not accumulated on the owning processes.

    """
    _L = _create_cpp_form(L)
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(_L),
         coeffs[1] if coeffs[1] is not None else pack_coefficients(_L))
    for b_sub, L_sub, constant, coeff in zip(b.getNestSubVecs(), _L, c[0], c[1]):
        with b_sub.localForm() as b_local:
            cpp.fem.assemble_vector(b_local.array_w, L_sub, constant, coeff)
    return b


# FIXME: Revise this interface
@functools.singledispatch
def assemble_vector_block(L: typing.List[typing.Union[Form, cpp.fem.Form]],
                          a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]],
                          constants_L=None, coefficients_L=None,
                          constants_a=None, coefficients_a=None,
                          bcs: typing.List[DirichletBC] = [],
                          x0: typing.Optional[PETSc.Vec] = None,
                          scale: float = 1.0) -> PETSc.Vec:
    """Assemble linear forms into a monolithic vector. The vector is not
    finalised, i.e. ghost values are not accumulated.

    """
    maps = [(form.function_spaces[0].dofmap.index_map, form.function_spaces[0].dofmap.index_map_bs)
            for form in _create_cpp_form(L)]
    b = cpp.fem.create_vector_block(maps)
    with b.localForm() as b_local:
        b_local.set(0.0)
    return assemble_vector_block(b, L, a, constants_L, coefficients_L, constants_a,
                                 coefficients_a, bcs, x0, scale)


@assemble_vector_block.register(PETSc.Vec)
def _(b: PETSc.Vec,
      L: typing.List[typing.Union[Form, cpp.fem.Form]],
      a,
      constants_L=None, coefficients_L=None,
      constants_a=None, coefficients_a=None,
      bcs: typing.List[DirichletBC] = [],
      x0: typing.Optional[PETSc.Vec] = None,
      scale: float = 1.0) -> PETSc.Vec:
    """Assemble linear forms into a monolithic vector. The vector is not
    zeroed and it is not finalised, i.e. ghost values are not
    accumulated.

    """
    maps = [(form.function_spaces[0].dofmap.index_map, form.function_spaces[0].dofmap.index_map_bs)
            for form in _create_cpp_form(L)]
    if x0 is not None:
        x0_local = cpp.la.get_local_vectors(x0, maps)
        x0_sub = x0_local
    else:
        x0_local = []
        x0_sub = [None] * len(maps)

    _L = _create_cpp_form(L)
    if constants_L is None:
        constants_L = pack_constants(_L)
    if coefficients_L is None:
        coefficients_L = pack_coefficients(_L)

    _a = _create_cpp_form(a)
    if constants_a is None:
        constants_a = pack_constants(_a)
    if coefficients_a is None:
        coefficients_a = pack_coefficients(_a)

    bcs1 = cpp.fem.bcs_cols(_create_cpp_form(a), bcs)
    b_local = cpp.la.get_local_vectors(b, maps)
    for b_sub, L_sub, a_sub, bc, constant_L, coeff_L, constant_a, coeff_a in zip(b_local, _L, _a, bcs1,
                                                                                 constants_L, coefficients_L,
                                                                                 constants_a, coefficients_a):
        cpp.fem.assemble_vector(b_sub, L_sub, constant_L, coeff_L)
        cpp.fem.apply_lifting(b_sub, a_sub, constant_a, coeff_a, bc, x0_local, scale)

    cpp.la.scatter_local_vectors(b, b_local, maps)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    bcs0 = cpp.fem.bcs_rows(_create_cpp_form(L), bcs)
    offset = 0
    b_array = b.getArray(readonly=False)
    for submap, bc, _x0 in zip(maps, bcs0, x0_sub):
        size = submap[0].size_local * submap[1]
        cpp.fem.set_bc(b_array[offset:offset + size], bc, _x0, scale)
        offset += size

    return b


# -- Matrix assembly ---------------------------------------------------------


@functools.singledispatch
def assemble_matrix(a: typing.Union[Form, cpp.fem.Form],
                    constants=None, coefficients=None,
                    bcs: typing.List[DirichletBC] = [],
                    diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    A = cpp.fem.create_matrix(_create_cpp_form(a))
    return assemble_matrix(A, a, constants, coefficients, bcs, diagonal)


@assemble_matrix.register(PETSc.Mat)
def _(A: PETSc.Mat,
      a: typing.Union[Form, cpp.fem.Form],
      constants=None, coefficients=None,
      bcs: typing.List[DirichletBC] = [],
      diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    _a = _create_cpp_form(a)
    if constants is None:
        constants = pack_constants(_a)
    if coefficients is None:
        coefficients = pack_coefficients(_a)
    cpp.fem.assemble_matrix_petsc(A, _a, constants, coefficients, bcs)
    if _a.function_spaces[0].id == _a.function_spaces[1].id:
        A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
        cpp.fem.insert_diagonal(A, _a.function_spaces[0], bcs, diagonal)
    return A


# FIXME: Revise this interface
@functools.singledispatch
def assemble_matrix_nest(a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]],
                         constants=None, coefficients=None,
                         bcs: typing.List[DirichletBC] = [], mat_types=[],
                         diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    A = cpp.fem.create_matrix_nest(_create_cpp_form(a), mat_types)
    assemble_matrix_nest(A, a, constants, coefficients, bcs, diagonal)
    return A


@assemble_matrix_nest.register(PETSc.Mat)
def _(A: PETSc.Mat,
      a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]],
      constants=None, coefficients=None,
      bcs: typing.List[DirichletBC] = [],
      diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    _a = _create_cpp_form(a)
    for i, a_row in enumerate(_a):
        for j, a_block in enumerate(a_row):
            if a_block is not None:
                Asub = A.getNestSubMatrix(i, j)
                const = (None if constants is None else constants[i][j])
                coeff = (None if coefficients is None else coefficients[i][j])
                assemble_matrix(Asub, a_block, const, coeff, bcs)
    return A


# FIXME: Revise this interface
@functools.singledispatch
def assemble_matrix_block(a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]],
                          constants=None, coefficients=None,
                          bcs: typing.List[DirichletBC] = [],
                          diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    A = cpp.fem.create_matrix_block(_create_cpp_form(a))
    return assemble_matrix_block(A, a, constants, coefficients, bcs, diagonal)


def _extract_function_spaces(a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]]):
    """From a rectangular array of bilinear forms, extraction the function spaces
    for each block row and block column

    """

    assert len({len(cols) for cols in a}) == 1, "Array of function spaces is not rectangular"

    # Extract (V0, V1) pair for each block in 'a'
    def fn(form):
        return form.function_spaces if form is not None else None
    from functools import partial
    Vblock = map(partial(map, fn), a)

    # Compute spaces for each row/column block
    rows = [set() for i in range(len(a))]
    cols = [set() for i in range(len(a[0]))]
    for i, Vrow in enumerate(Vblock):
        for j, V in enumerate(Vrow):
            if V is not None:
                rows[i].add(V[0])
                cols[j].add(V[1])

    rows = [e for row in rows for e in row]
    cols = [e for col in cols for e in col]
    assert len(rows) == len(a)
    assert len(cols) == len(a[0])
    return rows, cols


@assemble_matrix_block.register(PETSc.Mat)
def _(A: PETSc.Mat,
      a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]],
      constants=None, coefficients=None,
      bcs: typing.List[DirichletBC] = [],
      diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    _a = _create_cpp_form(a)
    if constants is None:
        constants = pack_constants(_a)
    if coefficients is None:
        coefficients = pack_coefficients(_a)

    V = _extract_function_spaces(_a)
    is_rows = cpp.la.create_petsc_index_sets([(Vsub.dofmap.index_map, Vsub.dofmap.index_map_bs) for Vsub in V[0]])
    is_cols = cpp.la.create_petsc_index_sets([(Vsub.dofmap.index_map, Vsub.dofmap.index_map_bs) for Vsub in V[1]])

    # Assemble form
    for i, a_row in enumerate(_a):
        for j, a_sub in enumerate(a_row):
            if a_sub is not None:
                Asub = A.getLocalSubMatrix(is_rows[i], is_cols[j])
                cpp.fem.assemble_matrix_petsc_unrolled(Asub, a_sub, constants[i][j], coefficients[i][j], bcs)
                A.restoreLocalSubMatrix(is_rows[i], is_cols[j], Asub)

    # Flush to enable switch from add to set in the matrix
    A.assemble(PETSc.Mat.AssemblyType.FLUSH)

    # Set diagonal
    for i, a_row in enumerate(_a):
        for j, a_sub in enumerate(a_row):
            if a_sub is not None:
                Asub = A.getLocalSubMatrix(is_rows[i], is_cols[j])
                if a_sub.function_spaces[0].id == a_sub.function_spaces[1].id:
                    cpp.fem.insert_diagonal(Asub, a_sub.function_spaces[0], bcs, diagonal)
                A.restoreLocalSubMatrix(is_rows[i], is_cols[j], Asub)

    return A


# -- Modifiers for Dirichlet conditions ---------------------------------------

def apply_lifting(b: PETSc.Vec,
                  a: typing.List[typing.Union[Form, cpp.fem.Form]],
                  bcs: typing.List[typing.List[DirichletBC]],
                  constants=None, coefficients=None,
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
    _a = _create_cpp_form(a)
    if constants is None:
        constants = pack_constants(_a)
    if coefficients is None:
        coefficients = pack_coefficients(_a)
    with contextlib.ExitStack() as stack:
        x0 = [stack.enter_context(x.localForm()) for x in x0]
        x0_r = [x.array_r for x in x0]
        b_local = stack.enter_context(b.localForm())
        cpp.fem.apply_lifting(b_local.array_w, _a, constants, coefficients, bcs, x0_r, scale)


def apply_lifting_nest(b: PETSc.Vec,
                       a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]],
                       bcs: typing.List[DirichletBC],
                       constants=None, coefficients=None,
                       x0: typing.Optional[PETSc.Vec] = None,
                       scale: float = 1.0) -> PETSc.Vec:
    """Modify nested vector for lifting of Dirichlet boundary conditions.

    """
    x0 = [] if x0 is None else x0.getNestSubVecs()
    _a = _create_cpp_form(a)
    if constants is None:
        constants = pack_constants(_a)
    if coefficients is None:
        coefficients = pack_coefficients(_a)
    bcs1 = cpp.fem.bcs_cols(_a, bcs)
    for b_sub, a_sub, constants, coeffs, bc1 in zip(b.getNestSubVecs(), _a, constants, coefficients, bcs1):
        apply_lifting(b_sub, a_sub, bc1, constants, coeffs, x0, scale)
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
    if x0 is not None:
        x0 = x0.array_r
    cpp.fem.set_bc(b.array_w, bcs, x0, scale)


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
    x0 = len(_b) * [None] if x0 is None else x0.getNestSubVecs()
    for b_sub, bc, x_sub in zip(_b, bcs, x0):
        set_bc(b_sub, bc, x_sub, scale)
