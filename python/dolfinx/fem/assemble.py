# Copyright (C) 2018-2026 Garth N. Wells, Jack S. Hale and Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Assembly functions for variational forms."""

from __future__ import annotations

import functools
import typing
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from dolfinx import cpp as _cpp
from dolfinx import default_scalar_type, la
from dolfinx.cpp.fem import pack_coefficients as _pack_coefficients
from dolfinx.cpp.fem import pack_constants as _pack_constants
from dolfinx.fem import IntegralType
from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.forms import Form
from dolfinx.fem.function import FunctionSpace
from dolfinx.fem.utils import create_sparsity_pattern
from dolfinx.typing import Scalar


@typing.overload
def pack_constants(form: None) -> None: ...


@typing.overload
def pack_constants(form: Form) -> npt.NDArray: ...


@typing.overload
def pack_constants(form: Sequence[Form]) -> list[npt.NDArray]: ...


@typing.overload
def pack_constants(form: Sequence[Form | None]) -> list[npt.NDArray | None]: ...


@typing.overload
def pack_constants(
    form: Sequence[Sequence[Form | None]],
) -> list[list[npt.NDArray | None]]: ...


def pack_constants(
    form: Form | Sequence[Form | None] | Sequence[Sequence[Form | None]] | None,
) -> npt.NDArray | Sequence[npt.NDArray | Sequence[npt.NDArray | None] | None] | None:
    """Pack form constants for use in assembly.

    Pack the 'constants' that appear in forms. The packed constants can
    then be passed to an assembler. This is a performance optimisation
    for cases where a form is assembled multiple times and (some)
    constants do not change.

    If ``form`` is a sequence of forms, this function returns an array
    of form constants with the same shape as ``form``.

    Args:
        form: Single form or sequence of forms to pack the constants
            for.

    Returns:
        A ``constant`` array for each form.
    """
    if form is None:
        return None
    elif isinstance(form, Sequence):
        return [pack_constants(f) for f in form]
    else:
        return _pack_constants(form._cpp_object)


@typing.overload
def pack_coefficients(form: Form | None) -> dict[tuple[IntegralType, int], npt.NDArray]: ...


@typing.overload
def pack_coefficients(
    form: Sequence[Form | None],
) -> list[dict[tuple[IntegralType, int], npt.NDArray]]: ...


@typing.overload
def pack_coefficients(
    form: Sequence[Sequence[Form | None]],
) -> list[list[dict[tuple[IntegralType, int], npt.NDArray]]]: ...


def pack_coefficients(
    form: Form | Sequence[Form | None] | Sequence[Sequence[Form | None]] | None,
) -> (
    dict[tuple[IntegralType, int], npt.NDArray]
    | Sequence[
        dict[tuple[IntegralType, int], npt.NDArray]
        | Sequence[dict[tuple[IntegralType, int], npt.NDArray]]
    ]
):
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
    if form is None:
        return {}
    elif isinstance(form, Sequence):
        return [pack_coefficients(f) for f in form]
    else:
        return _pack_coefficients(form._cpp_object)


# -- Vector and matrix instantiation --------------------------------------


def create_vector(V: FunctionSpace, dtype: npt.DTypeLike = default_scalar_type) -> la.Vector:
    """Create a Vector that is compatible with the given function space.

    Args:
        V: A function space.
        dtype: Data type of the vector.

    Returns:
        A vector compatible with the function space.
    """
    # Can just take the first dofmap here, since all dof maps have the same
    # index map in mixed-topology meshes
    dofmap = V.dofmaps[0]
    return la.vector(dofmap.index_map, dofmap.index_map_bs, dtype=dtype)


def create_matrix(a: Form, block_mode: la.BlockMode | None = None) -> la.MatrixCSR:
    """Create a sparse matrix that is compatible with a bilinear form.

    Args:
        a: Bilinear form.
        block_mode: Block mode of the CSR matrix. If ``None``, default
            is used.

    Returns:
        A sparse matrix that the form can be assembled into.
    """
    sp = create_sparsity_pattern(a)
    sp.finalize()
    if block_mode is not None:
        return la.matrix_csr(sp, block_mode=block_mode, dtype=a.dtype)
    else:
        return la.matrix_csr(sp, dtype=a.dtype)


# -- Scalar assembly ------------------------------------------------------


def assemble_scalar(
    M: Form,
    constants: npt.NDArray | None = None,
    coeffs: dict[tuple[IntegralType, int], npt.NDArray] | None = None,
) -> float | complex:
    """Assemble functional.

    The returned value is local and not accumulated across processes.

    Args:
        M: The functional to compute.
        constants: Constants that appear in the form. If ``None``, any
            required constants will be computed.
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
    if constants is None:
        constants = pack_constants(M)

    if coeffs is None:
        coeffs = pack_coefficients(M)

    return _cpp.fem.assemble_scalar(M._cpp_object, constants, coeffs)


# -- Vector assembly ------------------------------------------------------


@functools.singledispatch
def assemble_vector(
    L: typing.Any,
    constants: npt.NDArray | None = None,
    coeffs: dict[tuple[IntegralType, int], npt.NDArray] | None = None,
) -> la.Vector:
    """Assemble linear form into a vector."""
    return _assemble_vector_form(L, constants, coeffs)


@assemble_vector.register
def _assemble_vector_form(
    L: Form,
    constants: npt.NDArray | None = None,
    coeffs: dict[tuple[IntegralType, int], npt.NDArray] | None = None,
) -> la.Vector:
    """Assemble linear form into a new Vector.

    Args:
        L: The linear form to assemble.
        constants: Constants that appear in the form. If ``None``,
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
    b = create_vector(L.function_spaces[0], L.dtype)
    b.array[:] = 0

    if constants is None:
        constants = pack_constants(L)

    if coeffs is None:
        coeffs = pack_coefficients(L)

    _assemble_vector_array(b.array, L, constants, coeffs)
    return b


@assemble_vector.register(np.ndarray)
def _assemble_vector_array(
    b: npt.NDArray,
    L: Form,
    constants: npt.NDArray | None = None,
    coeffs: dict[tuple[IntegralType, int], npt.NDArray] | None = None,
) -> npt.NDArray:
    """Assemble linear form into an existing array.

    Args:
        b: Array to assemble the contribution from the calling MPI
            rank into. It must have the required size.
        L: Linear form assemble.
        constants: Constants that appear in the form. If ``None``,
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
    if constants is None:
        constants = pack_constants(L)

    if coeffs is None:
        coeffs = pack_coefficients(L)

    _cpp.fem.assemble_vector(b, L._cpp_object, constants, coeffs)
    return b


# -- Matrix assembly ------------------------------------------------------


def _unrolled_size(V: FunctionSpace) -> int:
    """Number of unrolled dofs of ``V``, owned plus ghost."""
    dofmap = V.dofmaps[0]
    imap = dofmap.index_map
    return dofmap.index_map_bs * (imap.size_local + imap.num_ghosts)


def _bc_dof_markers(V: FunctionSpace, bcs: Sequence[DirichletBC] | None) -> npt.NDArray[np.int8]:
    """Mark the dofs of ``V`` constrained by a boundary condition.

    Args:
        V: Space whose dofs (owned and ghost) are marked.
        bcs: Boundary conditions. Only those defined on ``V`` or a
            subspace of it contribute.

    Returns:
        Array with entry ``1`` for constrained dofs and ``0``
        otherwise, or an empty array if no boundary condition applies.
    """
    markers = None
    for bc in bcs or []:
        if V.contains(bc.function_space):
            if markers is None:
                markers = np.zeros(_unrolled_size(V), dtype=np.int8)
            markers[bc.dof_indices()[0]] = 1
    return np.empty(0, dtype=np.int8) if markers is None else markers


def _bc_dof_markers_by_space(
    spaces: Sequence[FunctionSpace | None], bcs: Sequence[DirichletBC] | None
) -> list[npt.NDArray[np.int8]]:
    """Constrained dof markers, one array per entry of ``spaces``.

    Each array has entry ``1`` for a constrained dof (owned and ghost,
    unrolled) and ``0`` otherwise. An entry is empty where the space is
    ``None`` or no boundary condition applies. Only conditions defined
    on a space or a subspace of it mark that space.

    Markers depend only on the space, so a space repeated in ``spaces``
    is marked once and the array shared: two entries may be the same
    array rather than equal copies. Callers must not modify them.
    """
    built: list[tuple[typing.Any, npt.NDArray[np.int8]]] = []
    markers = []
    for V in spaces:
        if V is None:
            markers.append(np.empty(0, dtype=np.int8))
            continue
        for space, m in built:
            if space is V._cpp_object:
                markers.append(m)
                break
        else:
            m = _bc_dof_markers(V, bcs)
            built.append((V._cpp_object, m))
            markers.append(m)
    return markers


def _bc_lifting_markers(
    spaces: Sequence[FunctionSpace | None],
    bcs: Sequence[Sequence[DirichletBC]],
) -> list[npt.NDArray[np.int8]]:
    """Constrained dof markers on each trial space.

    Entry ``j`` is ``1`` for the constrained dofs of ``spaces[j]``
    (owned and ghost), or empty if that space is ``None`` or has no
    boundary conditions. Unlike the values from
    :func:`_bc_lifting_values`, markers are fixed once the boundary
    conditions are built, so a repeated caller may reuse them.
    """
    return [
        np.empty(0, dtype=np.int8) if V is None else _bc_dof_markers(V, bcs0)
        for V, bcs0 in zip(spaces, bcs, strict=True)
    ]


def _bc_lifting_values(
    spaces: Sequence[FunctionSpace | None],
    bcs: Sequence[Sequence[DirichletBC]],
    dtype: npt.DTypeLike,
) -> list[npt.NDArray]:
    """Boundary condition values on each trial space.

    Entry ``j`` holds the values of ``bcs[j]`` where marked, as
    ``dtype``, or is empty if that space is ``None`` or has no boundary
    conditions. Where more than one condition constrains a dof, the
    last one in ``bcs[j]`` sets its value. Values must not be cached:
    the function or constant behind a condition may have changed since
    the last call.
    """
    values = []
    for V, bcs0 in zip(spaces, bcs, strict=True):
        if V is None or len(bcs0) == 0:
            values.append(np.empty(0, dtype=dtype))
            continue
        v = np.zeros(_unrolled_size(V), dtype=dtype)
        for bc in bcs0:
            bc.set(v, None, 1)
        values.append(v)
    return values


def _owned_marked_rows(V: FunctionSpace, markers: npt.NDArray[np.int8]) -> npt.NDArray[np.int32]:
    """Locally owned dofs of ``V`` that are marked in ``markers``."""
    if markers.size == 0:
        return np.empty(0, dtype=np.int32)
    dofmap = V.dofmaps[0]
    num_owned = dofmap.index_map_bs * dofmap.index_map.size_local
    return np.flatnonzero(markers[:num_owned]).astype(np.int32)


def _assemble_matrix_csr_markers(
    A: la.MatrixCSR,
    a: Form,
    dof_marker0: npt.NDArray[np.int8],
    dof_marker1: npt.NDArray[np.int8],
    diag: float = 1.0,
    constants: npt.NDArray | None = None,
    coeffs: dict[tuple[IntegralType, int], npt.NDArray] | None = None,
) -> la.MatrixCSR:
    """Assemble a bilinear form into a matrix, given constrained dofs.

    Rows marked in ``dof_marker0`` and columns marked in
    ``dof_marker1`` are zeroed. If the test and trial spaces are the
    same, ``diag`` is set on the diagonal of locally owned marked rows.
    See :func:`_bc_dof_markers` for the marker format; an empty array
    marks nothing.
    """
    if constants is None:
        constants = pack_constants(a)
    if coeffs is None:
        coeffs = pack_coefficients(a)

    V0, V1 = a.function_spaces
    _cpp.fem.assemble_matrix(  # type: ignore[arg-type]
        A._cpp_object,
        a._cpp_object,
        constants,
        coeffs,  # type: ignore[arg-type]
        dof_marker0,
        dof_marker1,
    )

    # If matrix is a 'diagonal' block, set diagonal entry for
    # constrained dofs. Assembly zeroed these rows, so adding sets it.
    if V0._cpp_object is V1._cpp_object:
        set_diagonal(A, _owned_marked_rows(V0, dof_marker0), diag, la.InsertMode.add)
    return A


@functools.singledispatch
def assemble_matrix(
    a: typing.Any,
    bcs: Sequence[DirichletBC] | None = None,
    diag: float = 1.0,
    constants: npt.NDArray | None = None,
    coeffs: dict[tuple[IntegralType, int], npt.NDArray] | None = None,
    block_mode: la.BlockMode | None = None,
) -> la.MatrixCSR:
    """Assemble bilinear form into a matrix.

    Args:
        a: The bilinear form assemble.
        bcs: Boundary conditions that affect the assembled matrix.
            Degrees-of-freedom constrained by a boundary condition will
            have their rows/columns zeroed and the value ``diag``
            set on the matrix diagonal.
        diag: Value to set on the matrix diagonal for Dirichlet
            boundary condition constrained degrees-of-freedom belonging
            to the same trial and test space.
        constants: Constants that appear in the form. If ``None``,
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

    Note:
        Convenience function for callers that have boundary conditions.
        It rebuilds the constrained dof markers on every call, and
        should not be called internally by the library.
    """
    A = create_matrix(a, block_mode)
    V0, V1 = a.function_spaces
    marker0, marker1 = _bc_dof_markers_by_space([V0, V1], bcs)
    _assemble_matrix_csr_markers(A, a, marker0, marker1, diag, constants, coeffs)
    return A


@assemble_matrix.register
def _assemble_matrix_csr(
    A: la.MatrixCSR,
    a: Form,
    bcs: Sequence[DirichletBC] | None = None,
    diag: float = 1.0,
    constants: npt.NDArray | None = None,
    coeffs: dict[tuple[IntegralType, int], npt.NDArray] | None = None,
) -> la.MatrixCSR:
    """Assemble bilinear form into a matrix.

    Args:
        A: The matrix to assemble into. It must have been initialized
            with the correct sparsity pattern.
        a: The bilinear form assemble.
        bcs: Boundary conditions that affect the assembled matrix.
            Degrees-of-freedom constrained by a boundary condition will
            have their rows/columns zeroed and the value ``diag``
            set on the diagonal.
        diag: Value to set on the matrix diagonal for Dirichlet
            boundary condition constrained degrees-of-freedom belonging
            to the same trial and test space.
        constants: Constants that appear in the form. If not provided,
            any required constants will be computed.
        coeffs: Coefficients that appear in the form. If not provided,
            any required coefficients will be computed.

    Returns:
        ``A``, for convenience.

    Note:
        The returned matrix is not finalised, i.e. ghost values are not
        accumulated.

    Note:
        Convenience function for callers that have boundary conditions.
        It rebuilds the constrained dof markers on every call, and
        should not be called internally by the library.
    """
    V0, V1 = a.function_spaces
    marker0, marker1 = _bc_dof_markers_by_space([V0, V1], bcs)
    return _assemble_matrix_csr_markers(A, a, marker0, marker1, diag, constants, coeffs)


def set_diagonal(
    A: la.MatrixCSR[Scalar],
    rows: npt.NDArray[np.int32],
    diagonal: Scalar | float | complex | npt.NDArray[Scalar] = 1.0,
    insert_mode: la.InsertMode = la.InsertMode.insert,
) -> None:
    """Set or add values on the diagonal for given rows of a matrix.

    Args:
        A: Matrix to modify.
        rows: Rows, in local indices, to set the diagonal value for.
        diagonal: Value to set on the diagonal, either a single value
            for all rows or an array with ``diagonal[i]`` the value for
            ``rows[i]``. An array must have the same length as
            ``rows``.
        insert_mode: ``la.InsertMode.insert`` to set the diagonal
            entries, or ``la.InsertMode.add`` to add to them.

    Note:
        A row that the calling rank does not own is accumulated into
        the owner's entry when the matrix is finalised, so pass owned
        rows unless that accumulation is intended. A row repeated in
        ``rows`` is likewise written once per occurrence.
    """
    if np.ndim(diagonal) > 0:
        diagonal = np.asarray(diagonal, dtype=A.data.dtype)
    typing.cast(typing.Any, _cpp.fem.set_diagonal)(A._cpp_object, rows, diagonal, insert_mode)


def set_bc_diagonal(
    A: la.MatrixCSR[Scalar],
    V: FunctionSpace,
    bcs: Sequence[DirichletBC[Scalar]] | None,
    diagonal: Scalar | float | complex = 1.0,
    insert_mode: la.InsertMode = la.InsertMode.insert,
) -> None:
    """Set a value on the diagonal of locally owned constrained rows.

    Only rows owned by the calling rank are set. A constrained
    degree-of-freedom that is a ghost here is left untouched and is
    set by the rank that owns it, so this function needs no
    communication.

    Args:
        A: Matrix to modify. Must be associated with ``V`` on both its
            row and column function spaces.
        V: Function space that the rows/columns of ``A`` are associated
            with.
        bcs: Boundary conditions that identify the diagonal rows to
            set. Only conditions defined on ``V`` or a subspace of it
            contribute, and of those only their locally owned dofs. If
            ``None``, no rows are set.
        diagonal: Value to set on the diagonal of each owned
            constrained row.
        insert_mode: ``la.InsertMode.insert`` to set the diagonal
            entries, or ``la.InsertMode.add`` to add to them.

    Note:
        Each row is set exactly once, even where several boundary
        conditions constrain the same degree-of-freedom, so
        ``la.InsertMode.add`` cannot double-count an overlap. Every
        condition sets the same ``diagonal`` value, so their order in
        ``bcs`` does not matter here.

    Note:
        Convenience interface for callers holding ``V`` and ``bcs``
        rather than the row list, which it rebuilds on every call.
        Library code passes the rows to :func:`set_diagonal` instead.
    """
    # Marking rather than concatenating makes the rows sorted and
    # duplicate-free by construction, so overlapping conditions need no
    # separate deduplication
    rows = _owned_marked_rows(V, _bc_dof_markers(V, bcs))
    set_diagonal(A, rows, diagonal, insert_mode)


def assemble_matrix_fn(
    fn: Callable[[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray], int],
    a: Form,
    bcs: Sequence[DirichletBC] | None = None,
) -> None:
    """Assemble a bilinear form, inserting element matrices via ``fn``.

    Rather than assembling into a :class:`~dolfinx.la.MatrixCSR` or a
    PETSc matrix, ``fn`` is called once per cell/facet contribution with
    the local-to-global row indices, column indices, and the element
    matrix values, and is responsible for inserting them into a
    caller-owned matrix representation.

    Args:
        fn: Called as ``fn(rows, cols, vals)`` for each contribution,
            where ``vals`` has shape ``(len(rows), len(cols))``. Return
            ``0`` on success.
        a: Bilinear form to assemble.
        bcs: Boundary conditions that affect the assembled matrix. Rows
            and columns constrained by a boundary condition are zeroed.

    Note:
        Convenience function for callers that have boundary conditions.
        It rebuilds the constrained dof markers on every call, and
        should not be called internally by the library.
    """
    V0, V1 = a.function_spaces
    marker0, marker1 = _bc_dof_markers_by_space([V0, V1], bcs)
    typing.cast(typing.Any, _cpp.fem.assemble_matrix)(
        fn,
        a._cpp_object,
        pack_constants(a),
        pack_coefficients(a),
        marker0,
        marker1,
    )


# -- Modifiers for Dirichlet conditions -----------------------------------


def apply_lifting(
    b: npt.NDArray,
    a: Sequence[Form],
    bcs: Sequence[Sequence[DirichletBC]],
    x0: Sequence[npt.NDArray] | None = None,
    alpha: float = 1,
    constants: Sequence[npt.NDArray] | None = None,
    coeffs: Sequence[dict[tuple[IntegralType, int], npt.NDArray]] | None = None,
) -> None:
    """Modify right-hand side for lifting of Dirichlet conditions.

    Consider the discrete algebraic system:

    .. math::

       \\begin{bmatrix} A_{0} & A_{1} \\end{bmatrix}
       \\begin{bmatrix}u_{0} \\\\ u_{1}\\end{bmatrix}
       = b,

    where :math:`A_{i}` is a matrix. Partitioning each vector
    :math:`u_{i}` into 'unknown' (:math:`u_{i}^{(0)}`) and prescribed
    (:math:`u_{i}^{(1)}`) groups,

    .. math::

        \\begin{bmatrix}
            A_{0}^{(0)} & A_{0}^{(1)} & A_{1}^{(0)} & A_{1}^{(1)}
        \\end{bmatrix}
        \\begin{bmatrix}
            u_{0}^{(0)} \\\\ u_{0}^{(1)} \\\\ u_{1}^{(0)} \\\\ u_{1}^{(1)}
        \\end{bmatrix}
        = b.

    If :math:`u_{i}^{(1)} = \\alpha(g_{i} - x_{i})`, where :math:`g_{i}`
    is the Dirichlet boundary condition value, :math:`x_{i}` is provided
    and :math:`\\alpha` is a constant, then

    .. math::

        \\begin{bmatrix}
            A_{0}^{(0)} & A_{0}^{(1)} & A_{1}^{(0)} & A_{1}^{(1)}
        \\end{bmatrix}
        \\begin{bmatrix}u_{0}^{(0)} \\\\ \\alpha(g_{0} - x_{0})
        \\\\ u_{1}^{(0)} \\\\ \\alpha(g_{1} - x_{1})\\end{bmatrix}
        = b.

    Rearranging,

    .. math::

        \\begin{bmatrix}A_{0}^{(0)} & A_{1}^{(0)}\\end{bmatrix}
        \\begin{bmatrix}u_{0}^{(0)} \\\\ u_{1}^{(0)}\\end{bmatrix}
        = b - \\alpha A_{0}^{(1)} (g_{0} - x_{0})
        - \\alpha A_{1}^{(1)} (g_{1} - x_{1}).

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
            generates the matrix :math:`A_{i}`. All forms in ``a`` must
            share the same test function space. The trial function
            spaces can differ.
        bcs: Boundary conditions that provide the :math:`g_{i}` values.
            ``bcs[i]`` is the sequence of boundary conditions on
            :math:`u_{i}`. Helper functions exist to build a
            list-of-lists of `DirichletBC` from a list of forms ``a``
            and a flat list of `DirichletBC` objects ``bcs``::

                bcs1 = fem.bcs_by_block(
                    fem.extract_function_spaces([a], 1),
                    bcs
                )

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

    Note:
        Convenience function for callers that have boundary conditions.
        It rebuilds the constrained dof markers and values on every
        call, and should not be called internally by the library.
    """  # noqa: D301
    spaces = [None if form is None else form.function_spaces[1] for form in a]
    _apply_lifting_markers(
        b,
        a,
        _bc_lifting_markers(spaces, bcs),
        _bc_lifting_values(spaces, bcs, b.dtype),
        x0,
        alpha,
        constants,
        coeffs,
    )


def _apply_lifting_markers(
    b: npt.NDArray,
    a: Sequence[Form | None],
    bc_markers1: Sequence[npt.NDArray[np.int8]],
    bc_values1: Sequence[npt.NDArray],
    x0: Sequence[npt.NDArray] | None = None,
    alpha: float = 1,
    constants: Sequence[npt.NDArray] | None = None,
    coeffs: Sequence[dict[tuple[IntegralType, int], npt.NDArray]] | None = None,
) -> None:
    """Lifting (see :func:`apply_lifting`), given constrained dofs.

    ``bc_markers1[j]`` and ``bc_values1[j]`` are the constrained dof
    markers and boundary condition values on the trial space of
    ``a[j]``, from :func:`_bc_lifting_markers` and
    :func:`_bc_lifting_values`. Empty arrays mean block ``j`` has no
    constraints.
    """
    if x0 is None:
        x0 = []

    if constants is None:
        constants = [
            pack_constants(form) if form is not None else np.array([], dtype=b.dtype) for form in a
        ]

    if coeffs is None:
        coeffs = [pack_coefficients(form) if form is not None else {} for form in a]

    _a = [None if form is None else form._cpp_object for form in a]
    _cpp.fem.apply_lifting(b, _a, constants, coeffs, bc_markers1, bc_values1, x0, alpha)  # type: ignore[arg-type]
