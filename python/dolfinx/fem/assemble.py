# Copyright (C) 2018-2021 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Assembly functions for variational forms."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from dolfinx.fem.forms import FormMetaClass

import collections
import functools

import numpy as np

import dolfinx
from dolfinx import cpp as _cpp
from dolfinx import la

# -- Packing constants and coefficients --------------------------------------


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


Coefficients = collections.namedtuple('Coefficients', ['constants', 'coeffs'])


# -- Vector instantiation ----------------------------------------------------

def create_vector(L: FormMetaClass, dtype=np.float64) -> la.VectorMetaClass:
    dofmap = L.function_spaces[0].dofmap
    return la.vector(dofmap.index_map, dofmap.index_map_bs, dtype)

# -- Matrix instantiation ----------------------------------------------------


def create_matrix(a: FormMetaClass, dtype=np.float64) -> la.MatrixCSRMetaClass:
    sp = dolfinx.fem.create_sparsity_pattern(a)
    sp.assemble()
    return la.matrix_csr(sp, dtype)


# -- Scalar assembly ---------------------------------------------------------


def assemble_scalar(M: FormMetaClass, coeffs=Coefficients(None, None)):
    """Assemble functional. The returned value is local and not
    accumulated across processes.

    """
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(M),
         coeffs[1] if coeffs[1] is not None else pack_coefficients(M))
    return _cpp.fem.assemble_scalar(M, c[0], c[1])


# -- Vector assembly ---------------------------------------------------------

@functools.singledispatch
def assemble_vector(L: FormMetaClass, coeffs=Coefficients(None, None)) -> la.VectorMetaClass:
    """Assemble linear form into a new Vector. The returned vector
    is not finalised, i.e. ghost values are not accumulated on the
    owning processes.

    """
    # TODO: get dtype from L
    b = la.vector(L.function_spaces[0].dofmap.index_map,
                  L.function_spaces[0].dofmap.index_map_bs)
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(L),
         coeffs[1] if coeffs[1] is not None else pack_coefficients(L))
    b.array[:] = 0
    _cpp.fem.assemble_vector(b.array, L, c[0], c[1])
    return b


@assemble_vector.register(np.ndarray)
def _(b: np.ndarray, L: FormMetaClass, coeffs=Coefficients(None, None)):
    """Assemble linear form into an existing PETSc vector. The vector is
    not zeroed before assembly and it is not finalised, i.e. ghost
    values are not accumulated on the owning processes.

    """
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(L),
         coeffs[1] if coeffs[1] is not None else pack_coefficients(L))
    _cpp.fem.assemble_vector(b, L, c[0], c[1])


# -- Matrix assembly ---------------------------------------------------------


# @functools.singledispatch
# def assemble_matrix(a: FormMetaClass, bcs: typing.List[DirichletBCMetaClass] = [],
#                     diagonal: float = 1.0,
#                     coeffs=Coefficients(None, None)):
#     """Assemble bilinear form into a matrix. The returned matrix is not
#     finalised, i.e. ghost values are not accumulated.

#     """
#     A = _cpp.fem.petsc.create_matrix(a)
#     return assemble_matrix(A, a, bcs, diagonal, coeffs)


# @assemble_matrix.register(PETSc.Mat)
# def _(A: PETSc.Mat,
#       a: FormMetaClass,
#       bcs: typing.List[DirichletBCMetaClass] = [],
#       diagonal: float = 1.0,
#       coeffs=Coefficients(None, None)) -> PETSc.Mat:
#     """Assemble bilinear form into a matrix. The returned matrix is not
#     finalised, i.e. ghost values are not accumulated.

#     """
#     c = (coeffs[0] if coeffs[0] is not None else pack_constants(a),
#          coeffs[1] if coeffs[1] is not None else pack_coefficients(a))
#     _cpp.fem.petsc.assemble_matrix(A, a, c[0], c[1], bcs)
#     if a.function_spaces[0].id == a.function_spaces[1].id:
#         A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
#         A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
#         _cpp.fem.petsc.insert_diagonal(A, a.function_spaces[0], bcs, diagonal)
#     return A


def _extract_function_spaces(a: typing.List[typing.List[FormMetaClass]]):
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


# -- Modifiers for Dirichlet conditions ---------------------------------------

# def apply_lifting(b: PETSc.Vec, a: typing.List[FormMetaClass],
#                   bcs: typing.List[typing.List[DirichletBCMetaClass]],
#                   x0: typing.Optional[typing.List[PETSc.Vec]] = [],
#                   scale: float = 1.0, coeffs=Coefficients(None, None)) -> None:
#     """Modify RHS vector b for lifting of Dirichlet boundary conditions.
#     It modifies b such that:

#         b <- b - scale * A_j (g_j - x0_j)

#     where j is a block (nest) index. For a non-blocked problem j = 0.
#     The boundary conditions bcs are on the trial spaces V_j. The forms
#     in [a] must have the same test space as L (from which b was built),
#     but the trial space may differ. If x0 is not supplied, then it is
#     treated as zero.

#     Ghost contributions are not accumulated (not sent to owner). Caller
#     is responsible for calling VecGhostUpdateBegin/End.
#     """
#     c = (coeffs[0] if coeffs[0] is not None else pack_constants(a),
#          coeffs[1] if coeffs[1] is not None else pack_coefficients(a))
#     with contextlib.ExitStack() as stack:
#         x0 = [stack.enter_context(x.localForm()) for x in x0]
#         x0_r = [x.array_r for x in x0]
#         b_local = stack.enter_context(b.localForm())
#         _cpp.fem.apply_lifting(b_local.array_w, a, c[0], c[1], bcs, x0_r, scale)


# def apply_lifting_nest(b: PETSc.Vec, a: typing.List[typing.List[FormMetaClass]],
#                        bcs: typing.List[DirichletBCMetaClass],
#                        x0: typing.Optional[PETSc.Vec] = None,
#                        scale: float = 1.0, coeffs=Coefficients(None, None)) -> PETSc.Vec:
#     """Modify nested vector for lifting of Dirichlet boundary conditions.

#     """
#     x0 = [] if x0 is None else x0.getNestSubVecs()
#     c = (coeffs[0] if coeffs[0] is not None else pack_constants(a),
#          coeffs[1] if coeffs[1] is not None else pack_coefficients(a))
#     bcs1 = bcs_by_block(extract_function_spaces(a, 1), bcs)
#     for b_sub, a_sub, constants, coeffs in zip(b.getNestSubVecs(), a, c[0], c[1]):
#         apply_lifting(b_sub, a_sub, bcs1, x0, scale, (constants, coeffs))
#     return b


# def set_bc(b: PETSc.Vec, bcs: typing.List[DirichletBCMetaClass],
#            x0: typing.Optional[PETSc.Vec] = None, scale: float = 1.0) -> None:
#     """Insert boundary condition values into vector. Only local (owned)
#     entries are set, hence communication after calling this function is
#     not required unless ghost entries need to be updated to the boundary
#     condition value.

#     """
#     if x0 is not None:
#         x0 = x0.array_r
#     _cpp.fem.set_bc(b.array_w, bcs, x0, scale)
