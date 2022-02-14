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
    from dolfinx.fem.bcs import DirichletBCMetaClass

import collections
import functools
import warnings

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


# -- Vector and matrix instantiation -----------------------------------------

def create_vector(L: FormMetaClass) -> la.VectorMetaClass:
    """Create a Vector that is compatible with a given linear form"""
    dofmap = L.function_spaces[0].dofmap
    return la.vector(dofmap.index_map, dofmap.index_map_bs, dtype=L.dtype)


def create_matrix(a: FormMetaClass) -> la.MatrixCSRMetaClass:
    """Create a sparse matrix that is compatible with a given bilinear form"""
    sp = dolfinx.fem.create_sparsity_pattern(a)
    sp.assemble()
    return la.matrix_csr(sp, dtype=a.dtype)


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
    b = create_vector(L)
    b.array[:] = 0
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(L),
         coeffs[1] if coeffs[1] is not None else pack_coefficients(L))
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
    return b

# -- Matrix assembly ---------------------------------------------------------


@functools.singledispatch
def assemble_matrix(a: FormMetaClass, bcs: typing.List[DirichletBCMetaClass] = [],
                    diagonal: float = 1.0,
                    coeffs=Coefficients(None, None)) -> la.MatrixCSRMetaClass:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    A = create_matrix(a)
    assemble_matrix(A, a, bcs, diagonal, coeffs)
    return A


@assemble_matrix.register(la.MatrixCSRMetaClass)
def _(A: la.MatrixCSRMetaClass, a: FormMetaClass,
      bcs: typing.List[DirichletBCMetaClass] = [],
      diagonal: float = 1.0, coeffs=Coefficients(None, None)) -> la.MatrixCSRMetaClass:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(a),
         coeffs[1] if coeffs[1] is not None else pack_coefficients(a))
    # _cpp.fem.assemble_matrix(A, a, c[0], c[1], bcs)
    _cpp.fem.assemble_matrix(A, a, c[0], c[1], bcs)

    # If matrix is a 'diagonal'block, set diagonal entry for constrained
    # dofs
    if a.function_spaces[0].id == a.function_spaces[1].id:
        if len(bcs) > 0:
            warnings.warn("Setting of matrix bc diagonals not yet implemented.")
    #     A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
    #     A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
    #     _cpp.fem.petsc.insert_diagonal(A, a.function_spaces[0], bcs, diagonal)
    return A


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

def apply_lifting(b: np.ndarray, a: typing.List[FormMetaClass],
                  bcs: typing.List[typing.List[DirichletBCMetaClass]],
                  x0: typing.Optional[typing.List[np.ndarray]] = [],
                  scale: float = 1.0, coeffs=Coefficients(None, None)) -> None:
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
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(a),
         coeffs[1] if coeffs[1] is not None else pack_coefficients(a))
    _cpp.fem.apply_lifting(b, a, c[0], c[1], bcs, x0, scale)


def set_bc(b: np.ndarray, bcs: typing.List[DirichletBCMetaClass],
           x0: typing.Optional[np.ndarray] = None, scale: float = 1.0) -> None:
    """Insert boundary condition values into vector. Only local (owned)
    entries are set, hence communication after calling this function is
    not required unless ghost entries need to be updated to the boundary
    condition value.

    """
    _cpp.fem.set_bc(b, bcs, x0, scale)
