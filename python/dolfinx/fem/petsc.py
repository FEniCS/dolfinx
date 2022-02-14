# Copyright (C) 2018-2021 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Assembly functions into PETSc objects for variational forms.

Functions in this module generally apply functions in :mod:`dolfinx.fem`
to PETSc linear algebra objects and handle any PETSc-specific
preparation."""


from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from dolfinx.fem.forms import FormMetaClass
    from dolfinx.fem.bcs import DirichletBCMetaClass

import collections
import contextlib
import functools

from dolfinx import cpp as _cpp
from dolfinx import la
from dolfinx.fem import assemble
from dolfinx.fem.assemble import _extract_function_spaces
from dolfinx.fem.bcs import bcs_by_block
from dolfinx.fem.forms import extract_function_spaces

from petsc4py import PETSc

Coefficients = collections.namedtuple('Coefficients', ['constants', 'coeffs'])


def pack_constants(form: typing.Union[FormMetaClass, typing.Sequence[FormMetaClass]]):
    """Compute form constants. If form is an array of forms, this
    function returns an array of form constants with the same shape as
    form.

    """
    def _pack(form):
        if form is None:
            return None
        elif isinstance(form, collections.abc.Iterable):
            return list(map(lambda sub_form: _pack(sub_form), form))
        else:
            return _cpp.fem.pack_constants(form)

    return _pack(form)


def pack_coefficients(form: typing.Union[FormMetaClass, typing.Sequence[FormMetaClass]]):
    """Compute form coefficients. If form is an array of forms, this
    function returns an array of form coefficients with the same shape
    as form.

    """
    def _pack(form):
        if form is None:
            return {}
        elif isinstance(form, collections.abc.Iterable):
            return list(map(lambda sub_form: _pack(sub_form), form))
        else:
            return _cpp.fem.pack_coefficients(form)

    return _pack(form)

# -- Vector instantiation ----------------------------------------------------


def create_vector(L: FormMetaClass) -> PETSc.Vec:
    """Create a PETSc vector that is compaible with a linear form.

    Args:
        L: A linear form.

    Returns:
        A PETSc vector with a layout that is compatible with `L`.

    """
    dofmap = L.function_spaces[0].dofmap
    return la.create_petsc_vector(dofmap.index_map, dofmap.index_map_bs)


def create_vector_block(L: typing.List[FormMetaClass]) -> PETSc.Vec:
    """Create a PETSc vector (blocked) that is compaible with a list of linear forms.

    Args:
        L: List of linear forms.

    Returns:
        A PETSc vector with a layout that is compatible with `L`.

    """
    maps = [(form.function_spaces[0].dofmap.index_map,
             form.function_spaces[0].dofmap.index_map_bs) for form in L]
    return _cpp.fem.petsc.create_vector_block(maps)


def create_vector_nest(L: typing.List[FormMetaClass]) -> PETSc.Vec:
    """Create a PETSc netsted vector (``VecNest``) that is compaible with a list of linear forms.

    Args:
        L: List of linear forms.

    Returns:
        A PETSc nested vector (``VecNest``) with a layout that is
        compatible with `L`.

    """
    maps = [(form.function_spaces[0].dofmap.index_map,
             form.function_spaces[0].dofmap.index_map_bs) for form in L]
    return _cpp.fem.petsc.create_vector_nest(maps)


# -- Matrix instantiation ----------------------------------------------------

def create_matrix(a: FormMetaClass, mat_type=None) -> PETSc.Mat:
    """Create a PETSc matrix that is compaible with a bilinear form.

    Args:
        a: A bilinear form.
        mat_type: The PETSc vector (``Vec``) type.

    Returns:
        A PETSc matrix with a layout that is compatible with `a`.

    """
    if mat_type is None:
        return _cpp.fem.petsc.create_matrix(a)
    else:
        return _cpp.fem.petsc.create_matrix(a, mat_type)


def create_matrix_block(a: typing.List[typing.List[FormMetaClass]]) -> PETSc.Mat:
    """Create a PETSc matrix that is compaible with a rectangular array of bilinear forms.

    Args:
        a: A rectangular array of bilinear forms.

    Returns:
        A PETSc matrix with a blocked layout that is compatible with `a`.

    """
    return _cpp.fem.petsc.create_matrix_block(a)


def create_matrix_nest(a: typing.List[typing.List[FormMetaClass]]) -> PETSc.Mat:
    """Create a PETSc matrix (``MatNest``) that is compaible with a rectangular array of bilinear forms.

    Args:
        a: A rectangular array of bilinear forms.

    Returns:
        A PETSc matrix ('MatNest``) that is compatible with `a`.

    """
    return _cpp.fem.petsc.create_matrix_nest(a)


# -- Vector assembly ---------------------------------------------------------

@functools.singledispatch
def assemble_vector(L: FormMetaClass, coeffs=Coefficients(None, None)) -> PETSc.Vec:
    """Assemble linear form into a new PETSc vector.

    Note:
        The returned vector is not finalised, i.e. ghost values are not
        accumulated on the owning processes.

    Args:
        L: A linear form.

    Returns:
        An assembled vector.

    """
    b = la.create_petsc_vector(L.function_spaces[0].dofmap.index_map,
                               L.function_spaces[0].dofmap.index_map_bs)
    with b.localForm() as b_local:
        assemble.assemble_vector(b_local.array_w, L, coeffs[0], coeffs[1])
    return b


@assemble_vector.register(PETSc.Vec)
def _(b: PETSc.Vec, L: FormMetaClass, coeffs=Coefficients(None, None)) -> PETSc.Vec:
    """Assemble linear form into an existing PETSc vector.

    Note:
        The vector is not zeroed before assembly and it is not
        finalised, i.e. ghost values are not accumulated on the owning
        processes.

    Args:
        b: Vector to assemble the contribution of the linear form into.
        L: A linear form to assemble into `b`.

    Returns:
        An assembled vector.

    """
    with b.localForm() as b_local:
        assemble.assemble_vector(b_local.array_w, L, coeffs)
    return b


@functools.singledispatch
def assemble_vector_nest(L: FormMetaClass, coeffs=Coefficients(None, None)) -> PETSc.Vec:
    """Assemble linear forms into a new nested PETSc (VecNest) vector.
    The returned vector is not finalised, i.e. ghost values are not
    accumulated on the owning processes.

    """
    maps = [(form.function_spaces[0].dofmap.index_map,
             form.function_spaces[0].dofmap.index_map_bs) for form in L]
    b = _cpp.fem.petsc.create_vector_nest(maps)
    for b_sub in b.getNestSubVecs():
        with b_sub.localForm() as b_local:
            b_local.set(0.0)
    return assemble_vector_nest(b, L, coeffs)


@assemble_vector_nest.register(PETSc.Vec)
def _(b: PETSc.Vec, L: typing.List[FormMetaClass], constants=None, coefficients=None) -> PETSc.Vec:
    """Assemble linear forms into a nested PETSc (VecNest) vector. The
    vector is not zeroed before assembly and it is not finalised, i.e.
    ghost values are not accumulated on the owning processes.

    """
    # c = (coeffs[0] if coeffs[0] is not None else pack_constants(L),
    #      coeffs[1] if coeffs[1] is not None else pack_coefficients(L))
    constants = constants or [_cpp.fem.pack_constants(form) for form in L]
    coefficients = coefficients or [_cpp.fem.pack_coefficients(form) for form in L]
    for b_sub, L_sub, const, coeff in zip(b.getNestSubVecs(), L, constants, coefficients):
        with b_sub.localForm() as b_local:
            assemble.assemble_vector(b_local.array_w, L_sub, const, coeff)
    return b


# FIXME: Revise this interface
@functools.singledispatch
def assemble_vector_block(L: typing.List[FormMetaClass],
                          a: typing.List[typing.List[FormMetaClass]],
                          bcs: typing.List[DirichletBCMetaClass] = [],
                          x0: typing.Optional[PETSc.Vec] = None,
                          scale: float = 1.0,
                          coeffs_L=Coefficients(None, None),
                          coeffs_a=Coefficients(None, None)) -> PETSc.Vec:
    """Assemble linear forms into a monolithic vector. The vector is not
    finalised, i.e. ghost values are not accumulated.

    """
    maps = [(form.function_spaces[0].dofmap.index_map,
             form.function_spaces[0].dofmap.index_map_bs) for form in L]
    b = _cpp.fem.petsc.create_vector_block(maps)
    with b.localForm() as b_local:
        b_local.set(0.0)
    return assemble_vector_block(b, L, a, bcs, x0, scale, coeffs_L, coeffs_a)


@assemble_vector_block.register(PETSc.Vec)
def _(b: PETSc.Vec,
      L: typing.List[FormMetaClass],
      a,
      bcs: typing.List[DirichletBCMetaClass] = [],
      x0: typing.Optional[PETSc.Vec] = None,
      scale: float = 1.0,
      coeffs_L=Coefficients(None, None),
      coeffs_a=Coefficients(None, None)) -> PETSc.Vec:
    """Assemble linear forms into a monolithic vector. The vector is not
    zeroed and it is not finalised, i.e. ghost values are not
    accumulated.

    """
    maps = [(form.function_spaces[0].dofmap.index_map,
             form.function_spaces[0].dofmap.index_map_bs) for form in L]
    if x0 is not None:
        x0_local = _cpp.la.petsc.get_local_vectors(x0, maps)
        x0_sub = x0_local
    else:
        x0_local = []
        x0_sub = [None] * len(maps)

    c_L = (coeffs_L[0] if coeffs_L[0] is not None else pack_constants(L),
           coeffs_L[1] if coeffs_L[1] is not None else pack_coefficients(L))
    c_a = (coeffs_a[0] if coeffs_a[0] is not None else pack_constants(a),
           coeffs_a[1] if coeffs_a[1] is not None else pack_coefficients(a))

    bcs1 = bcs_by_block(extract_function_spaces(a, 1), bcs)
    b_local = _cpp.la.petsc.get_local_vectors(b, maps)
    for b_sub, L_sub, a_sub, constant_L, coeff_L, constant_a, coeff_a in zip(b_local, L, a,
                                                                             c_L[0], c_L[1],
                                                                             c_a[0], c_a[1]):
        _cpp.fem.assemble_vector(b_sub, L_sub, constant_L, coeff_L)
        _cpp.fem.apply_lifting(b_sub, a_sub, constant_a, coeff_a, bcs1, x0_local, scale)

    _cpp.la.petsc.scatter_local_vectors(b, b_local, maps)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    bcs0 = bcs_by_block(extract_function_spaces(L), bcs)
    offset = 0
    b_array = b.getArray(readonly=False)
    for submap, bc, _x0 in zip(maps, bcs0, x0_sub):
        size = submap[0].size_local * submap[1]
        _cpp.fem.set_bc(b_array[offset: offset + size], bc, _x0, scale)
        offset += size

    return b


# -- Matrix assembly ---------------------------------------------------------


@functools.singledispatch
def assemble_matrix(a: FormMetaClass, bcs: typing.List[DirichletBCMetaClass] = [],
                    diagonal: float = 1.0,
                    coeffs=Coefficients(None, None)) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    A = _cpp.fem.petsc.create_matrix(a)
    assemble_matrix(A, a, bcs, diagonal, coeffs)
    return A


@assemble_matrix.register(PETSc.Mat)
def _(A: PETSc.Mat, a: FormMetaClass, bcs: typing.List[DirichletBCMetaClass] = [],
      diagonal: float = 1.0, coeffs=Coefficients(None, None)) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(a),
         coeffs[1] if coeffs[1] is not None else pack_coefficients(a))
    _cpp.fem.petsc.assemble_matrix(A, a, c[0], c[1], bcs)
    if a.function_spaces[0].id == a.function_spaces[1].id:
        A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
        _cpp.fem.petsc.insert_diagonal(A, a.function_spaces[0], bcs, diagonal)
    return A


# FIXME: Revise this interface
@functools.singledispatch
def assemble_matrix_nest(a: typing.List[typing.List[FormMetaClass]],
                         bcs: typing.List[DirichletBCMetaClass] = [], mat_types=[],
                         diagonal: float = 1.0,
                         coeffs=Coefficients(None, None)) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    A = _cpp.fem.petsc.create_matrix_nest(a, mat_types)
    assemble_matrix_nest(A, a, bcs, diagonal, coeffs)
    return A


@assemble_matrix_nest.register(PETSc.Mat)
def _(A: PETSc.Mat, a: typing.List[typing.List[FormMetaClass]],
      bcs: typing.List[DirichletBCMetaClass] = [], diagonal: float = 1.0,
      coeffs=Coefficients(None, None)) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(a),
         coeffs[1] if coeffs[1] is not None else pack_coefficients(a))
    for i, (a_row, const_row, coeff_row) in enumerate(zip(a, c[0], c[1])):
        for j, (a_block, const, coeff) in enumerate(zip(a_row, const_row, coeff_row)):
            if a_block is not None:
                Asub = A.getNestSubMatrix(i, j)
                assemble_matrix(Asub, a_block, bcs, diagonal, (const, coeff))
    return A


# FIXME: Revise this interface
@functools.singledispatch
def assemble_matrix_block(a: typing.List[typing.List[FormMetaClass]],
                          bcs: typing.List[DirichletBCMetaClass] = [],
                          diagonal: float = 1.0,
                          coeffs=Coefficients(None, None)) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    A = _cpp.fem.petsc.create_matrix_block(a)
    return assemble_matrix_block(A, a, bcs, diagonal, coeffs)


@assemble_matrix_block.register(PETSc.Mat)
def _(A: PETSc.Mat, a: typing.List[typing.List[FormMetaClass]],
      bcs: typing.List[DirichletBCMetaClass] = [], diagonal: float = 1.0,
      coeffs=Coefficients(None, None)) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(a),
         coeffs[1] if coeffs[1] is not None else pack_coefficients(a))

    V = _extract_function_spaces(a)
    is_rows = _cpp.la.petsc.create_index_sets([(Vsub.dofmap.index_map, Vsub.dofmap.index_map_bs) for Vsub in V[0]])
    is_cols = _cpp.la.petsc.create_index_sets([(Vsub.dofmap.index_map, Vsub.dofmap.index_map_bs) for Vsub in V[1]])

    # Assemble form
    for i, a_row in enumerate(a):
        for j, a_sub in enumerate(a_row):
            if a_sub is not None:
                Asub = A.getLocalSubMatrix(is_rows[i], is_cols[j])
                _cpp.fem.petsc.assemble_matrix(Asub, a_sub, c[0][i][j], c[1][i][j], bcs, True)
                A.restoreLocalSubMatrix(is_rows[i], is_cols[j], Asub)

    # Flush to enable switch from add to set in the matrix
    A.assemble(PETSc.Mat.AssemblyType.FLUSH)

    # Set diagonal
    for i, a_row in enumerate(a):
        for j, a_sub in enumerate(a_row):
            if a_sub is not None:
                Asub = A.getLocalSubMatrix(is_rows[i], is_cols[j])
                if a_sub.function_spaces[0].id == a_sub.function_spaces[1].id:
                    _cpp.fem.petsc.insert_diagonal(Asub, a_sub.function_spaces[0], bcs, diagonal)
                A.restoreLocalSubMatrix(is_rows[i], is_cols[j], Asub)

    return A


# -- Modifiers for Dirichlet conditions ---------------------------------------

def apply_lifting(b: PETSc.Vec, a: typing.List[FormMetaClass],
                  bcs: typing.List[typing.List[DirichletBCMetaClass]],
                  x0: typing.Optional[typing.List[PETSc.Vec]] = [],
                  scale: float = 1.0, constants=None, coefficients=None) -> None:
    """Apply the function :func:`dolfinx.fem.apply_lifting` to a PETSc Vector."""
    with contextlib.ExitStack() as stack:
        x0 = [stack.enter_context(x.localForm()) for x in x0]
        x0_r = [x.array_r for x in x0]
        b_local = stack.enter_context(b.localForm())
        assemble.apply_lifting(b_local.array_w, a, bcs, x0_r, scale, constants, coefficients)


def apply_lifting_nest(b: PETSc.Vec, a: typing.List[typing.List[FormMetaClass]],
                       bcs: typing.List[DirichletBCMetaClass],
                       x0: typing.Optional[PETSc.Vec] = None,
                       scale: float = 1.0, constants=None, coefficients=None) -> PETSc.Vec:
    """Apply the function :func:`dolfinx.fem.apply_lifting` to each sub-vector in a nested PETSc Vector."""

    x0 = [] if x0 is None else x0.getNestSubVecs()
    bcs1 = bcs_by_block(extract_function_spaces(a, 1), bcs)

    constants = constants or [[form and _cpp.fem.pack_constants(form) for form in forms] for forms in a]
    coefficients = coefficients or [[{} if form is None else _cpp.fem.pack_coefficients(form) for form in forms] for forms in a]
    for b_sub, a_sub, const, coeff in zip(b.getNestSubVecs(), a, constants, coefficients):
        apply_lifting(b_sub, a_sub, bcs1, x0, scale, const, coeff)
    return b


def set_bc(b: PETSc.Vec, bcs: typing.List[DirichletBCMetaClass],
           x0: typing.Optional[PETSc.Vec] = None, scale: float = 1.0) -> None:
    """Apply the function :func:`dolfinx.fem.set_bc` to a PETSc Vector."""
    if x0 is not None:
        x0 = x0.array_r
    assemble.set_bc(b.array_w, bcs, x0, scale)


def set_bc_nest(b: PETSc.Vec, bcs: typing.List[typing.List[DirichletBCMetaClass]],
                x0: typing.Optional[PETSc.Vec] = None, scale: float = 1.0) -> None:
    """Apply the function :func:`dolfinx.fem.set_bc` to each sub-vector of a nested PETSc Vector."""
    _b = b.getNestSubVecs()
    x0 = len(_b) * [None] if x0 is None else x0.getNestSubVecs()
    for b_sub, bc, x_sub in zip(_b, bcs, x0):
        set_bc(b_sub, bc, x_sub, scale)
