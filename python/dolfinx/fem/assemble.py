# Copyright (C) 2018-2022 Garth N. Wells, Jack S. Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Assembly functions for variational forms."""

from __future__ import annotations

import collections
import functools
import typing

import numpy as np

import dolfinx
from dolfinx import cpp as _cpp
from dolfinx import la
from dolfinx.cpp.fem import pack_coefficients as _pack_coefficients
from dolfinx.cpp.fem import pack_constants as _pack_constants
from dolfinx.fem.bcs import DirichletBCMetaClass
from dolfinx.fem.forms import FormMetaClass, form_types


def pack_constants(form: typing.Union[FormMetaClass,
                                      typing.Sequence[FormMetaClass]]) -> typing.Union[np.ndarray,
                                                                                       typing.Sequence[np.ndarray]]:
    """Compute form constants.

    Pack the `constants` that appear in forms. The packed constants can
    be passed to an assembler. This is a performance optimisation for
    cases where a form is assembled multiple times and (some) constants
    do not change.

    If ``form`` is an array of forms, this function returns an array of
    form constants with the same shape as form.

    Args:
        form: A single form or array of forms to pack the constants for.

    Returns:
        A `constant` array for each form.

    """
    def _pack(form):
        if form is None:
            return None
        elif isinstance(form, collections.abc.Iterable):
            return list(map(lambda sub_form: _pack(sub_form), form))
        else:
            return _pack_constants(form)

    return _pack(form)


def pack_coefficients(form: typing.Union[FormMetaClass, typing.Sequence[FormMetaClass]]):
    """Compute form coefficients.

    Pack the `coefficients` that appear in forms. The packed
    coefficients can be passed to an assembler. This is a
    performance optimisation for cases where a form is assembled
    multiple times and (some) coefficients do not change.

    If ``form`` is an array of forms, this function returns an array of
    form coefficients with the same shape as form.

    Args:
        form: A single form or array of forms to pack the constants for.

    Returns:
        Coefficients for each form.

    """
    def _pack(form):
        if form is None:
            return {}
        elif isinstance(form, collections.abc.Iterable):
            return list(map(lambda sub_form: _pack(sub_form), form))
        else:
            return _pack_coefficients(form)

    return _pack(form)

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

def assemble_scalar(M: FormMetaClass, constants=None, coeffs=None):
    """Assemble functional. The returned value is local and not
    accumulated across processes.

    Args:
        M: The functional to compute.
        constants: Constants that appear in the form. If not provided,
            any required constants will be computed.
        coeffs: Coefficients that appear in the form. If not provided,
            any required coefficients will be computed.

    Return:
        The computed scalar on the calling rank.

    Note:
        Passing `constants` and `coefficients` is a performance
        optimisation for when a form is assembled multiple times and
        when (some) constants and coefficients are unchanged.

        To compute the functional value on the whole domain, the output
        of this function is typically summed across all MPI ranks.

    """
    constants = constants or _pack_constants(M)
    coeffs = coeffs or _pack_coefficients(M)
    return _cpp.fem.assemble_scalar(M, constants, coeffs)


# -- Vector assembly ---------------------------------------------------------

@functools.singledispatch
def assemble_vector(L: typing.Any,
                    constants=None, coeffs=None):
    return _assemble_vector_form(L, constants, coeffs)


@assemble_vector.register(FormMetaClass)
def _assemble_vector_form(L: form_types, constants=None, coeffs=None) -> la.VectorMetaClass:
    """Assemble linear form into a new Vector.

    Args:
        L: The linear form to assemble.
        constants: Constants that appear in the form. If not provided,
            any required constants will be computed.
        coeffs: Coefficients that appear in the form. If not provided,
            any required coefficients will be computed.

    Return:
        The assembled vector for the calling rank.

    Note:
        Passing `constants` and `coefficients` is a performance
        optimisation for when a form is assembled multiple times and
        when (some) constants and coefficients are unchanged.

    Note:
        The returned vector is not finalised, i.e. ghost values are not
        accumulated on the owning processes. Calling
        :func:`dolfinx.la.VectorMetaClass.scatter_reverse` on the
        return vector can accumulate ghost contributions.

    """
    b = create_vector(L)
    b.array[:] = 0
    constants = constants or _pack_constants(L)
    coeffs = coeffs or _pack_coefficients(L)
    _assemble_vector_array(b.array, L, constants, coeffs)
    return b


@assemble_vector.register(np.ndarray)
def _assemble_vector_array(b: np.ndarray, L: FormMetaClass, constants=None, coeffs=None):
    """Assemble linear form into a new Vector.

    Args:
        b: The array to assemble the contribution from the calling MPI
            rank into. It must have the required size.
        L: The linear form assemble.
        constants: Constants that appear in the form. If not provided,
            any required constants will be computed.
        coeffs: Coefficients that appear in the form. If not provided,
            any required coefficients will be computed.

    Note:
        Passing `constants` and `coefficients` is a performance
        optimisation for when a form is assembled multiple times and
        when (some) constants and coefficients are unchanged.

    Note:
        The returned vector is not finalised, i.e. ghost values are not
        accumulated on the owning processes. Calling
        :func:`dolfinx.la.VectorMetaClass.scatter_reverse` on the
        return vector can accumulate ghost contributions.

    """

    constants = _pack_constants(L) if constants is None else constants
    coeffs = _pack_coefficients(L) if coeffs is None else coeffs
    _cpp.fem.assemble_vector(b, L, constants, coeffs)
    return b

# -- Matrix assembly ---------------------------------------------------------


@functools.singledispatch
def assemble_matrix(a: typing.Any,
                    bcs: typing.Optional[typing.List[DirichletBCMetaClass]] = None,
                    diagonal: float = 1.0, constants=None, coeffs=None):
    return _assemble_matrix_form(a, bcs, diagonal, constants, coeffs)


@assemble_matrix.register
def _assemble_matrix_csr(A: la.MatrixCSRMetaClass, a: form_types,
                         bcs: typing.Optional[typing.List[DirichletBCMetaClass]] = None,
                         diagonal: float = 1.0, constants=None, coeffs=None) -> la.MatrixCSRMetaClass:
    """Assemble bilinear form into a matrix.

        Args:
        a: The bilinear form assemble.
        bcs: Boundary conditions that affect the assembled matrix.
            Degrees-of-freedom constrained by a boundary condition will
            have their rows/columns zeroed and the value ``diagonal`` set
            on on the matrix diagonal.
        constants: Constants that appear in the form. If not provided,
            any required constants will be computed.
        coeffs: Coefficients that appear in the form. If not provided,
            any required coefficients will be computed.

    Note:
        The returned matrix is not finalised, i.e. ghost values are not
        accumulated.

    """
    bcs = [] if bcs is None else bcs
    constants = _pack_constants(a) if constants is None else constants
    coeffs = _pack_coefficients(a) if coeffs is None else coeffs
    _cpp.fem.assemble_matrix(A, a, constants, coeffs, bcs)

    # If matrix is a 'diagonal'block, set diagonal entry for constrained
    # dofs
    if a.function_spaces[0] is a.function_spaces[1]:
        _cpp.fem.insert_diagonal(A, a.function_spaces[0], bcs, diagonal)
    return A


@assemble_matrix.register(FormMetaClass)
def _assemble_matrix_form(a: form_types, bcs: typing.Optional[typing.List[DirichletBCMetaClass]] = None,
                          diagonal: float = 1.0,
                          constants=None, coeffs=None) -> la.MatrixCSRMetaClass:
    """Assemble bilinear form into a matrix.

    Args:
        a: The bilinear form assemble.
        bcs: Boundary conditions that affect the assembled matrix.
            Degrees-of-freedom constrained by a boundary condition will
            have their rows/columns zeroed and the value ``diagonal``
            set on on the matrix diagonal.
        constants: Constants that appear in the form. If not provided,
            any required constants will be computed.
        coeffs: Coefficients that appear in the form. If not provided,
            any required coefficients will be computed.

    Returns:
        Matrix representation of the bilinear form ``a``.

    Note:
        The returned matrix is not finalised, i.e. ghost values are not
        accumulated.

    """
    bcs = [] if bcs is None else bcs
    A: la.MatrixCSRMetaClass = create_matrix(a)
    _assemble_matrix_csr(A, a, bcs, diagonal, constants, coeffs)
    return A


# -- Modifiers for Dirichlet conditions ---------------------------------------


def apply_lifting(b: np.ndarray, a: typing.List[FormMetaClass],
                  bcs: typing.List[typing.List[DirichletBCMetaClass]],
                  x0: typing.Optional[typing.List[np.ndarray]] = None,
                  scale: float = 1.0, constants=None, coeffs=None) -> None:
    """Modify RHS vector b for lifting of Dirichlet boundary conditions.

    It modifies b such that:

    .. math::

        b \\leftarrow  b - \\text{scale} * A_j (g_j - x0_j)

    where j is a block (nest) index. For a non-blocked problem j = 0.
    The boundary conditions bcs are on the trial spaces V_j. The forms
    in [a] must have the same test space as L (from which b was built),
    but the trial space may differ. If x0 is not supplied, then it is
    treated as zero.

    Note:
        Ghost contributions are not accumulated (not sent to owner).
        Caller is responsible for calling VecGhostUpdateBegin/End.

    """
    x0 = [] if x0 is None else x0
    constants = [form and _pack_constants(form) for form in a] if constants is None else constants
    coeffs = [{} if form is None else _pack_coefficients(form) for form in a] if coeffs is None else coeffs
    _cpp.fem.apply_lifting(b, a, constants, coeffs, bcs, x0, scale)


def set_bc(b: np.ndarray, bcs: typing.List[DirichletBCMetaClass],
           x0: typing.Optional[np.ndarray] = None, scale: float = 1.0) -> None:
    """Insert boundary condition values into vector. Only local (owned)
    entries are set, hence communication after calling this function is
    not required unless ghost entries need to be updated to the boundary
    condition value.

    """
    _cpp.fem.set_bc(b, bcs, x0, scale)
