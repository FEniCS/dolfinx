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
import warnings
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

import dolfinx
from dolfinx import cpp as _cpp
from dolfinx import la
from dolfinx.cpp.fem import pack_coefficients as _pack_coefficients
from dolfinx.cpp.fem import pack_constants as _pack_constants
from dolfinx.fem import IntegralType
from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.forms import Form


def pack_constants(
    form: typing.Union[Form, typing.Sequence[Form]],
) -> typing.Union[np.ndarray, typing.Sequence[np.ndarray]]:
    """Pack form constants for use in assembly.

    Pack the 'constants' that appear in forms. The packed constants can
    then be passed to an assembler. This is a performance optimisation
    for cases where a form is assembled multiple times and (some)
    constants do not change.

    If ``form`` is a sequence of forms, this function returns an array
    of form constants with the same shape as ``form``.

    Args:
        form: Single form or sequence  of forms to pack the constants for.

    Returns:
        A ``constant`` array for each form.
    """

    def _pack(form):
        if form is None:
            return None
        elif isinstance(form, collections.abc.Iterable):
            return list(map(lambda sub_form: _pack(sub_form), form))
        else:
            return _pack_constants(form._cpp_object)

    return _pack(form)


def pack_coefficients(
    form: typing.Union[Form, typing.Sequence[Form]],
) -> typing.Union[
    dict[tuple[IntegralType, int], npt.NDArray], list[dict[tuple[IntegralType, int], npt.NDArray]]
]:
    """Pack form coefficients for use in assembly.

    Pack the ``coefficients`` that appear in forms. The packed
    coefficients can be passed to an assembler. This is a performance
    optimisation for cases where a form is assembled multiple times and
    (some) coefficients do not change.

    If ``form`` is an array of forms, this function returns an array of
    form coefficients with the same shape as ``form``.

    Args:
        form: A form or a sequence of forms to pack the coefficients
        for.

    Returns:
        Coefficients for each form.
    """

    def _pack(form):
        if form is None:
            return {}
        elif isinstance(form, collections.abc.Iterable):
            return list(map(lambda sub_form: _pack(sub_form), form))
        else:
            return _pack_coefficients(form._cpp_object)

    return _pack(form)


# -- Vector and matrix instantiation -----------------------------------------


def create_vector(L: Form) -> la.Vector:
    """Create a Vector that is compatible with a given linear form.

    Args:
        L: A linear form.

    Returns:
        A vector that the form can be assembled into.
    """
    # Can just take the first dofmap here, since all dof maps have the same
    # index map in mixed-topology meshes
    dofmap = L.function_spaces[0].dofmaps(0)
    return la.vector(dofmap.index_map, dofmap.index_map_bs, dtype=L.dtype)


def create_matrix(a: Form, block_mode: typing.Optional[la.BlockMode] = None) -> la.MatrixCSR:
    """Create a sparse matrix that is compatible with a given bilinear form.

    Args:
        a: Bilinear form.
        block_mode: Block mode of the CSR matrix. If ``None``, default
            is used.

    Returns:
        A sparse matrix that the form can be assembled into.
    """
    sp = dolfinx.fem.create_sparsity_pattern(a)
    sp.finalize()
    if block_mode is not None:
        return la.matrix_csr(sp, block_mode=block_mode, dtype=a.dtype)
    else:
        return la.matrix_csr(sp, dtype=a.dtype)


# -- Scalar assembly ---------------------------------------------------------


def assemble_scalar(M: Form, constants=None, coeffs=None):
    """Assemble functional. The returned value is local and not
    accumulated across processes.

    Args:
        M: The functional to compute.
        constants: Constants that appear in the form. If not provided,
            any required constants will be computed.
        coeffs: Coefficients that appear in the form. If not provided,
            any required coefficients will be computed.

    Returns:
        The computed scalar on the calling rank.

    Note:
        Passing `constants` and `coefficients` is a performance
        optimisation for when a form is assembled multiple times and
        when (some) constants and coefficients are unchanged.

        To compute the functional value on the whole domain, the output
        of this function is typically summed across all MPI ranks.
    """
    constants = constants or pack_constants(M)
    coeffs = coeffs or pack_coefficients(M)
    return _cpp.fem.assemble_scalar(M._cpp_object, constants, coeffs)


# -- Vector assembly ---------------------------------------------------------


@functools.singledispatch
def assemble_vector(L: typing.Any, constants=None, coeffs=None):
    return _assemble_vector_form(L, constants, coeffs)


@assemble_vector.register(Form)
def _assemble_vector_form(L: Form, constants=None, coeffs=None) -> la.Vector:
    """Assemble linear form into a new Vector.

    Args:
        L: The linear form to assemble.
        constants: Constants that appear in the form. If not provided,
            any required constants will be computed.
        coeffs: Coefficients that appear in the form. If not provided,
            any required coefficients will be computed.

    Returns:
        The assembled vector for the calling rank.

    Note:
        Passing `constants` and `coefficients` is a performance
        optimisation for when a form is assembled multiple times and
        when (some) constants and coefficients are unchanged.

    Note:
        The returned vector is not finalised, i.e. ghost values are not
        accumulated on the owning processes. Calling
        :func:`dolfinx.la.Vector.scatter_reverse` on the return vector
        can accumulate ghost contributions.
    """
    b = create_vector(L)
    b.array[:] = 0
    constants = constants or pack_constants(L)
    coeffs = coeffs or pack_coefficients(L)
    _assemble_vector_array(b.array, L, constants, coeffs)
    return b


@assemble_vector.register(np.ndarray)
def _assemble_vector_array(b: np.ndarray, L: Form, constants=None, coeffs=None):
    """Assemble linear form into an existing array.

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
        :func:`dolfinx.la.Vector.scatter_reverse` on the return vector
        can accumulate ghost contributions.
    """
    constants = pack_constants(L) if constants is None else constants
    coeffs = pack_coefficients(L) if coeffs is None else coeffs
    _cpp.fem.assemble_vector(b, L._cpp_object, constants, coeffs)
    return b


# -- Matrix assembly ---------------------------------------------------------


@functools.singledispatch
def assemble_matrix(
    a: typing.Any,
    bcs: typing.Optional[list[DirichletBC]] = None,
    diagonal: float = 1.0,
    constants=None,
    coeffs=None,
    block_mode: typing.Optional[la.BlockMode] = None,
):
    """Assemble bilinear form into a matrix.

    Args:
        a: The bilinear form assemble.
        bcs: Boundary conditions that affect the assembled matrix.
            Degrees-of-freedom constrained by a boundary condition will
            have their rows/columns zeroed and the value ``diagonal``
            set on on the matrix diagonal.
        diagonal: Value to set on the matrix diagonal for Dirichlet
            boundary condition constrained degrees-of-freedom belonging
            to the same trial and test space.
        constants: Constants that appear in the form. If not provided,
            any required constants will be computed.
        coeffs: Coefficients that appear in the form. If not provided,
            any required coefficients will be computed.
         block_mode: Block size mode for the returned space matrix. If
            ``None``, default is used.

    Returns:
        Matrix representation of the bilinear form ``a``.

    Note:
        The returned matrix is not finalised, i.e. ghost values are not
        accumulated.
    """
    bcs = [] if bcs is None else bcs
    A: la.MatrixCSR = create_matrix(a, block_mode)
    _assemble_matrix_csr(A, a, bcs, diagonal, constants, coeffs)
    return A


@assemble_matrix.register
def _assemble_matrix_csr(
    A: la.MatrixCSR,
    a: Form,
    bcs: typing.Optional[list[DirichletBC]] = None,
    diagonal: float = 1.0,
    constants=None,
    coeffs=None,
) -> la.MatrixCSR:
    """Assemble bilinear form into a matrix.

    Args:
        A: The matrix to assemble into. It must have been initialized
            with the correct sparsity pattern.
        a: The bilinear form assemble.
        bcs: Boundary conditions that affect the assembled matrix.
            Degrees-of-freedom constrained by a boundary condition will
            have their rows/columns zeroed and the value ``diagonal``
            set on the diagonal.
        diagonal: Value to set on the matrix diagonal for Dirichlet
            boundary condition constrained degrees-of-freedom belonging
            to the same trial and test space.
        constants: Constants that appear in the form. If not provided,
            any required constants will be computed. the matrix
            diagonal.
        coeffs: Coefficients that appear in the form. If not provided,
            any required coefficients will be computed.

    Note:
        The returned matrix is not finalised, i.e. ghost values are not
        accumulated.
    """
    bcs = [] if bcs is None else [bc._cpp_object for bc in bcs]
    constants = pack_constants(a) if constants is None else constants
    coeffs = pack_coefficients(a) if coeffs is None else coeffs
    _cpp.fem.assemble_matrix(A._cpp_object, a._cpp_object, constants, coeffs, bcs)

    # If matrix is a 'diagonal'block, set diagonal entry for constrained
    # dofs
    if a.function_spaces[0] is a.function_spaces[1]:
        _cpp.fem.insert_diagonal(A._cpp_object, a.function_spaces[0], bcs, diagonal)
    return A


# -- Modifiers for Dirichlet conditions ---------------------------------------


def apply_lifting(
    b: np.ndarray,
    a: Iterable[Form],
    bcs: Iterable[Iterable[DirichletBC]],
    x0: typing.Optional[Iterable[np.ndarray]] = None,
    alpha: float = 1,
    constants=None,
    coeffs=None,
) -> None:
    """Modify right-hand side vector ``b`` for lifting of Dirichlet boundary conditions.

    Consider the discrete algebraic system:

    .. math::

       \\begin{bmatrix} A_{0} & A_{1} \\end{bmatrix}
       \\begin{bmatrix}u_{0} \\\\ u_{1}\\end{bmatrix}
       = b,

    where :math:`A_{i}` is a matrix. Partitioning each vector
    :math:`u_{i}` into 'unknown' (:math:`u_{i}^{(0)}`) and prescribed
    (:math:`u_{i}^{(1)}`) groups,

    .. math::

        \\begin{bmatrix} A_{0}^{(0)} & A_{0}^{(1)} & A_{1}^{(0)} & A_{1}^{(1)}\\end{bmatrix}
        \\begin{bmatrix}u_{0}^{(0)} \\\\ u_{0}^{(1)}
        \\\\ u_{1}^{(0)} \\\\ u_{1}^{(1)}\\end{bmatrix}
        = b.

    If :math:`u_{i}^{(1)} = \\alpha(g_{i} - x_{i})`, where :math:`g_{i}`
    is the Dirichlet boundary condition value, :math:`x_{i}` is provided
    and :math:`\\alpha` is a constant, then

    .. math::

        \\begin{bmatrix}A_{0}^{(0)} & A_{0}^{(1)} & A_{1}^{(0)} & A_{1}^{(1)} \\end{bmatrix}
        \\begin{bmatrix}u_{0}^{(0)} \\\\ \\alpha(g_{0} - x_{0})
        \\\\ u_{1}^{(0)} \\\\ \\alpha(g_{1} - x_{1})\\end{bmatrix}
        = b.

    Rearranging,

    .. math::

        \\begin{bmatrix}A_{0}^{(0)} & A_{1}^{(0)}\\end{bmatrix}
        \\begin{bmatrix}u_{0}^{(0)} \\\\ u_{1}^{(0)}\\end{bmatrix}
        = b - \\alpha A_{0}^{(1)} (g_{0} - x_{0}) - \\alpha A_{1}^{(1)} (g_{1} - x_{1}).

    The modified  :math:`b` vector is

    .. math::

        b \\leftarrow b - \\alpha A_{0}^{(1)} (g_{0} - x_{0})
        - \\alpha A_{1}^{(1)} (g_{1} - x_{1})

    More generally,

    .. math::
        b \\leftarrow b - \\alpha A_{i}^{(1)} (g_{i} - x_{i}).

    Args:
        b: The array to modify inplace.
        a: List of bilinear forms, where ``a[i]`` is the form that
            generates the matrix :math"`A_{i}`. All forms in ``a`` must
            share the same test function space. The trial function
            spaces can differ.
        bcs: Boundary conditions that provide the :math:`g_{i}` values.
            ``bcs1[i]`` is the sequence of boundary conditions on
            :math:`u_{i}`.
        x0: The array :math:`x_{i}` above. If ``None`` it is set to
            zero.
        alpha: Scalar used in the modification of ``b``.
        constants: Packed constant data appearing in the forms ``a``. If
            ``None``, the constant data will be packed by the function.
        coeffs: Packed coefficient data appearing in the forms ``a``. If
            ``None``, the coefficient data will be packed by the
            function.

    Note:
        Ghost contributions are not accumulated (not sent to owner).
        Caller is responsible for reverse-scatter to update the ghosts.

    Note:
        Boundary condition values are *not* set in ``b`` by this
        function. Use :func:`dolfinx.fem.DirichletBC.set` to set values
        in ``b``.
    """
    x0 = [] if x0 is None else x0
    constants = (
        [pack_constants(form) if form is not None else np.array([], dtype=b.dtype) for form in a]
        if constants is None
        else constants
    )
    coeffs = (
        [{} if form is None else pack_coefficients(form) for form in a]
        if coeffs is None
        else coeffs
    )
    _a = [None if form is None else form._cpp_object for form in a]
    _bcs = [[bc._cpp_object for bc in bcs0] for bcs0 in bcs]
    _cpp.fem.apply_lifting(b, _a, constants, coeffs, _bcs, x0, alpha)


def set_bc(
    b: np.ndarray,
    bcs: list[DirichletBC],
    x0: typing.Optional[np.ndarray] = None,
    scale: float = 1,
) -> None:
    """Insert boundary condition values into vector.

    Note:
        This function is deprecated.

    Only local (owned) entries are set, hence communication after
    calling this function is not required unless ghost entries need to
    be updated to the boundary condition value.
    """
    warnings.warn(
        "dolfinx.fem.assembler.set_bc is deprecated. Use dolfinx.fem.DirichletBC.set instead.",
        DeprecationWarning,
    )
    for bc in bcs:
        bc.set(b, x0, scale)
