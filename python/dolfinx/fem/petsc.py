# Copyright (C) 2018-2026 Garth N. Wells, Nathan Sime and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""High-level solver classes and functions for assembling PETSc objects.

Functions in this module generally apply functions in :mod:`dolfinx.fem`
to PETSc linear algebra objects and handle any PETSc-specific
preparation.

Note:
    The following does not apply to the high-level classes
    :class:`dolfinx.fem.petsc.LinearProblem`
    :class:`dolfinx.fem.petsc.NonlinearProblem`.

    Due to subtle issues in the interaction between petsc4py memory
    management and the Python garbage collector, it is recommended that
    the PETSc method ``destroy()`` is called on returned PETSc objects
    once the object is no longer required. Note that ``destroy()`` is
    collective over the object's MPI communicator.
"""

from __future__ import annotations

import contextlib
import ctypes as _ctypes
import functools
import os
import pathlib
import typing
from collections.abc import Sequence
from typing import overload

from petsc4py import PETSc

import dolfinx
from dolfinx.log import LogLevel, log

if not dolfinx.has_petsc4py:
    raise RuntimeError("DOLFINx has not been built with petsc4py support.")


import numpy as np
from numpy import typing as npt

import dolfinx.cpp as _cpp
import dolfinx.la.petsc
import ufl
from dolfinx.common import IndexMap
from dolfinx.cpp.fem.petsc import discrete_curl as _discrete_curl
from dolfinx.cpp.fem.petsc import discrete_gradient as _discrete_gradient
from dolfinx.cpp.fem.petsc import interpolation_matrix as _interpolation_matrix
from dolfinx.fem import pack_coefficients, pack_constants
from dolfinx.fem.assemble import (
    _apply_lifting_markers,
    _assemble_vector_array,
    _bc_dof_markers_by_space,
    _bc_lifting_markers,
    _bc_lifting_values,
    _owned_marked_rows,
)
from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.bcs import bcs_by_block as _bcs_by_block
from dolfinx.fem.forms import Form, derivative_block
from dolfinx.fem.forms import extract_function_spaces as _extract_function_spaces
from dolfinx.fem.forms import form as _create_form
from dolfinx.fem.function import Function as _Function
from dolfinx.fem.function import FunctionSpace as _FunctionSpace
from dolfinx.mesh import EntityMap as _EntityMap

__all__ = [
    "LinearProblem",
    "NonlinearProblem",
    "apply_lifting",
    "assemble_jacobian",
    "assemble_matrix",
    "assemble_residual",
    "assemble_vector",
    "assign",
    "cffi_utils",
    "create_matrix",
    "create_vector",
    "ctypes_utils",
    "discrete_curl",
    "discrete_gradient",
    "interpolation_matrix",
    "numba_utils",
    "set_bc",
]


# -- Vector instantiation -------------------------------------------------


def create_vector(
    V: _FunctionSpace | Sequence[_FunctionSpace | None],
    /,
    kind: str | None = None,
) -> PETSc.Vec:
    """Create a vector compatible with linear form(s) or function space(s).

    Three cases are supported:

    1. For a single space ``V``, if ``kind`` is ``None`` or is
       ``PETSc.Vec.Type.MPI``, a ghosted PETSc vector which is compatible
       with ``V`` is created.

    2. If ``V`` is a sequence of functionspaces and ``kind`` is ``None`` or
       is ``PETSc.Vec.Type.MPI``, a ghosted PETSc vector which is
       compatible with ``V`` is created. The created vector ``b``
       is initialized such that on each MPI process ``b = [b_0, b_1, ...,
       b_n, b_0g, b_1g, ..., b_ng]``, where ``b_i`` are the entries
       associated with the 'owned' degrees-of-freedom for ``V[i]`` and
       ``b_ig`` are the 'unowned' (ghost) entries for ``V[i]``.

       For this case, the returned vector has an attribute ``_blocks``
       that holds the local offsets into ``b`` for the (i) owned and
       (ii) ghost entries for each ``V_i``. It can be accessed by
       ``b.getAttr("_blocks")``. The offsets can be used to get views
       into ``b`` for blocks, e.g.::

           >>> offsets0, offsets1, = b.getAttr("_blocks")
           >>> offsets0
           (0, 12, 28)
           >>> offsets1
           (28, 32, 35)
           >>> b0_owned = b.array[offsets0[0]:offsets0[1]]
           >>> b0_ghost = b.array[offsets1[0]:offsets1[1]]
           >>> b1_owned = b.array[offsets0[1]:offsets0[2]]
           >>> b1_ghost = b.array[offsets1[1]:offsets1[2]]

    3. If ``L/V`` is a sequence of linear forms/functionspaces and ``kind``
       is ``PETSc.Vec.Type.NEST``, a PETSc nested vector (a 'nest' of
       ghosted PETSc vectors) which is compatible with ``L/V`` is created.

    Args:
        V: Function space or a sequence of such.
        kind: PETSc vector type (``VecType``) to create.

    Returns:
        A PETSc vector with a layout that is compatible with ``V``. The
        vector is not initialised to zero.
    """
    if isinstance(V, _FunctionSpace):
        V = [V]
    elif any(_V is None for _V in V):
        raise RuntimeError("Can not create vector for None block.")

    maps = [(_V.dofmap.index_map, _V.dofmap.index_map_bs) for _V in V]  # type: ignore
    return dolfinx.la.petsc.create_vector(maps, kind=kind)


def _create_vector_from_form(L: Form | Sequence[Form], kind: str | None = None) -> PETSc.Vec:
    """Create a vector from the function spaces of linear forms."""
    spaces = typing.cast(
        _FunctionSpace | Sequence[_FunctionSpace | None], _extract_function_spaces(L)
    )
    return create_vector(spaces, kind=kind)


# -- Matrix instantiation -------------------------------------------------


def create_matrix(
    a: Form | Sequence[Sequence[Form | None]],
    kind: str | Sequence[Sequence[str | None]] | None = None,
) -> PETSc.Mat:
    """Create a matrix compatible with a sequence of bilinear forms.

    Three cases are supported:

    1. For a single bilinear form, it creates a compatible PETSc matrix
       of type ``kind``.
    2. For a rectangular array of bilinear forms, if ``kind`` is
       ``PETSc.Mat.Type.NEST`` or ``kind`` is an array of PETSc ``Mat``
       types (with the same shape as ``a``), a matrix of type
       ``PETSc.Mat.Type.NEST`` is created. The matrix is compatible
       with the forms ``a``.
    3. For a rectangular array of bilinear forms, it create a single
       (non-nested) matrix of type ``kind`` that is compatible with the
       array of for forms ``a``. If ``kind`` is ``None`` or
       ``PETSc.Vec.Type.MPI``, then the matrix is the default type.

       In this case, the matrix is arranged::

             A = [a_00 ... a_0n]
                 [a_10 ... a_1n]
                 [     ...     ]
                 [a_m0 ..  a_mn]

    Args:
        a: A bilinear form or a nested sequence of bilinear forms. Each
            block row and column must use a distinct function space; use
            ``FunctionSpace.clone()`` for separate blocks on the same
            finite element space.
        kind: The PETSc matrix type (``MatType``). An entry of an array
            of types is unused, and may be ``None``, where ``a`` holds
            no form.

    Returns:
        A PETSc matrix.

    Raises:
        ValueError: If ``kind`` is ``PETSc.Mat.Type.IS`` (MATIS) and the
            mesh has ghost cells. MATIS requires each process to hold
            one non-overlapping subdomain; build the mesh with
            ``GhostMode.none``.
    """
    if isinstance(a, Sequence):
        _extract_block_spaces(a)
        _a = [[None if form is None else form._cpp_object for form in arow] for arow in a]
        if kind == PETSc.Mat.Type.NEST:
            # Create nest matrix with default types
            return _cpp.fem.petsc.create_matrix_nest(_a, None)  # type: ignore[arg-type]
        else:
            if kind is None or isinstance(kind, str):  # Single 'kind' type
                # "mpi" is create_vector's sentinel, not a Mat type
                mat_kind = None if kind == PETSc.Vec.Type.MPI else kind
                return _cpp.fem.petsc.create_matrix_block(_a, mat_kind)  # type: ignore[arg-type]
            else:  # Array of 'kind' types
                return _cpp.fem.petsc.create_matrix_nest(_a, kind)  # type: ignore[arg-type]
    else:  # Single form
        return _cpp.fem.petsc.create_matrix(a._cpp_object, kind)  # type: ignore


# -- Vector assembly ------------------------------------------------------
@overload
def assemble_vector(
    L: Form | Sequence[Form],
    constants: npt.NDArray | Sequence[npt.NDArray] | None = None,
    coeffs: (
        dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]
        | Sequence[dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]]
        | None
    ) = None,
    kind: str | None = None,
) -> PETSc.Vec: ...


@overload
def assemble_vector(
    b: PETSc.Vec,
    L: Form | Sequence[Form],
    constants: npt.NDArray | Sequence[npt.NDArray] | None = None,
    coeffs: (
        dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]
        | Sequence[dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]]
        | None
    ) = None,
) -> PETSc.Vec: ...


@functools.singledispatch
def assemble_vector(
    L: Form | Sequence[Form],
    constants: npt.NDArray | Sequence[npt.NDArray] | None = None,
    coeffs: (
        dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]
        | Sequence[dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]]
        | None
    ) = None,
    kind: str | None = None,
) -> PETSc.Vec:
    """Assemble linear form(s) into a new PETSc vector.

    Three cases are supported:

    1. If ``L`` is a single linear form, the form is assembled into a
       ghosted PETSc vector.

    2. If ``L`` is a sequence of linear forms and ``kind`` is ``None``
       or is ``PETSc.Vec.Type.MPI``, the forms are assembled into a
       vector ``b`` such that ``b = [b_0, b_1, ..., b_n, b_0g, b_1g,
       ..., b_ng]`` where ``b_i`` are the entries associated with the
       'owned' degrees-of-freedom for ``L[i]`` and ``b_ig`` are the
       'unowned' (ghost) entries for ``L[i]``.

       For this case, the returned vector has an attribute ``_blocks``
       that holds the local offsets into ``b`` for the (i) owned and
       (ii) ghost entries for each ``L[i]``. See :func:`create_vector`
       for a description of the offset blocks.

    3. If ``L`` is a sequence of linear forms and ``kind`` is
       ``PETSc.Vec.Type.NEST``, the forms are assembled into a PETSc
       nested vector ``b`` (a nest of ghosted PETSc vectors) such that
       ``L[i]`` is assembled into the ith nested matrix in ``b``.

    Constant and coefficient data that appear in the forms(s) can be
    packed outside of this function to avoid re-packing by this
    function. The functions :func:`dolfinx.fem.pack_constants` and
    :func:`dolfinx.fem.pack_coefficients` can be used to 'pre-pack' the
    data.

    Note:
        The returned vector is not finalised, i.e. ghost values are not
        accumulated on the owning processes.

    Args:
        L: A linear form or sequence of linear forms.
        constants: Constants appearing in the form. For a single form,
            ``constants.ndim==1``. For multiple forms, the constants for
            form ``L[i]`` are  ``constants[i]``.
        coeffs: Coefficients appearing in the form. For a single form,
            ``coeffs.shape=(num_cells, n)``. For multiple forms, the
            coefficients for form ``L[i]`` are  ``coeffs[i]``.
        kind: PETSc vector type.

    Returns:
        An assembled vector.
    """
    b = _create_vector_from_form(L, kind=kind)
    dolfinx.la.petsc._zero_vector(b)
    return typing.cast(PETSc.Vec, _assemble_vector_petsc(b, L, constants, coeffs))


@assemble_vector.register  # type: ignore[attr-defined]
def _assemble_vector_petsc(
    b: PETSc.Vec,
    L: Form | Sequence[Form],
    constants: npt.NDArray | Sequence[npt.NDArray] | None = None,
    coeffs: (
        dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]
        | Sequence[dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]]
        | None
    ) = None,
) -> PETSc.Vec:
    """Assemble linear form(s) into a PETSc vector.

    The vector ``b`` must have been initialized with a size/layout that
    is consistent with the linear form. The vector ``b`` is normally
    created by :func:`create_vector`.

    Constants and coefficients that appear in the forms(s) can be passed
    to avoid re-computation of constants and coefficients. The functions
    :func:`dolfinx.fem.assemble.pack_constants` and
    :func:`dolfinx.fem.assemble.pack_coefficients` can be called.

    Note:
        The vector is not zeroed before assembly and it is not
        finalised, i.e. ghost values are not accumulated on the owning
        processes.

    Args:
        b: Vector to assemble the contribution of the linear form into.
        L: A linear form or sequence of linear forms to assemble into
            ``b``.
        constants: Constants appearing in the form. For a single form,
            ``constants.ndim==1``. For multiple forms, the constants for
            form ``L[i]`` are  ``constants[i]``.
        coeffs: Coefficients appearing in the form. For a single form,
            ``coeffs.shape=(num_cells, n)``. For multiple forms, the
            coefficients for form ``L[i]`` are  ``coeffs[i]``.

    Returns:
        Assembled vector.
    """
    if b.getType() == PETSc.Vec.Type.NEST:
        if not isinstance(L, Sequence):
            raise ValueError("Must provide a sequence of forms when assembling a nest vector")
        if isinstance(constants, np.ndarray):
            raise ValueError("Must provide a sequence of constants when assembling a nest vector")
        if isinstance(coeffs, dict):
            raise ValueError(
                "Must provide a sequence of coefficients when assembling a nest vector"
            )
        constants_nest: Sequence[npt.NDArray | None] = (
            [None] * len(L) if constants is None else constants
        )
        coeffs_nest: Sequence[dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray] | None] = (
            [None] * len(L) if coeffs is None else coeffs
        )
        for b_sub, L_sub, const, coeff in zip(
            b.getNestSubVecs(), L, constants_nest, coeffs_nest, strict=True
        ):
            assert L_sub is not None
            with b_sub.localForm() as b_local:
                _assemble_vector_array(b_local.array_w, L_sub, const, coeff)
    elif isinstance(L, Sequence):
        if isinstance(constants, np.ndarray):
            raise ValueError("Must provide a sequence of constants when assembling blocked forms")
        if isinstance(coeffs, dict):
            raise ValueError(
                "Must provide a sequence of coefficients when assembling blocked forms"
            )
        constants_block: Sequence[npt.NDArray] = (
            pack_constants(L)
            if constants is None
            else typing.cast(Sequence[npt.NDArray], constants)
        )
        coeffs_block: Sequence[dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]] = (
            pack_coefficients(L)
            if coeffs is None
            else typing.cast(
                Sequence[dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]], coeffs
            )
        )
        offset0, offset1 = b.getAttr("_blocks")  # type: ignore
        with b.localForm() as b_l:
            for L_, const, coeff, off0, off1, offg0, offg1 in zip(
                L,
                constants_block,
                coeffs_block,
                offset0[:-1],
                offset0[1:],
                offset1[:-1],
                offset1[1:],
                strict=True,
            ):
                bx_ = np.zeros((off1 - off0) + (offg1 - offg0), dtype=PETSc.ScalarType)
                _assemble_vector_array(bx_, L_, const, coeff)
                size = off1 - off0
                b_l.array_w[off0:off1] += bx_[:size]
                b_l.array_w[offg0:offg1] += bx_[size:]
    else:
        if isinstance(constants, Sequence) or isinstance(coeffs, Sequence):
            raise ValueError(
                "Must not provide a sequence of constants/coefficients for a single form"
            )
        with b.localForm() as b_local:
            _assemble_vector_array(b_local.array_w, L, constants, coeffs)

    return b


# -- Matrix assembly ------------------------------------------------------
@overload
def assemble_matrix(
    a: Form | Sequence[Sequence[Form | None]],
    bcs: Sequence[DirichletBC] | None = None,
    diag: float = 1.0,
    constants: npt.NDArray | Sequence[Sequence[npt.NDArray]] | None = None,
    coeffs: dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]
    | Sequence[Sequence[dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]]]
    | None = None,
    kind: str | Sequence[Sequence[str | None]] | None = None,
) -> PETSc.Mat: ...


@overload
def assemble_matrix(
    A: PETSc.Mat,
    a: Form | Sequence[Sequence[Form | None]],
    bcs: Sequence[DirichletBC] | None = None,
    diag: float = 1.0,
    constants: npt.NDArray | Sequence[Sequence[npt.NDArray]] | None = None,
    coeffs: (
        dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]
        | Sequence[Sequence[dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]]]
        | None
    ) = None,
) -> PETSc.Mat: ...


@functools.singledispatch
def assemble_matrix(
    a: Form | Sequence[Sequence[Form | None]],
    bcs: Sequence[DirichletBC] | None = None,
    diag: float = 1,
    constants: npt.NDArray | Sequence[Sequence[npt.NDArray]] | None = None,
    coeffs: (
        dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]
        | Sequence[Sequence[dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]]]
        | None
    ) = None,
    kind: str | Sequence[Sequence[str | None]] | None = None,
) -> PETSc.Mat:
    """Assemble a bilinear form into a matrix.

    The following cases are supported:

    1. If ``a`` is a single bilinear form, the form is assembled
       into PETSc matrix of type ``kind``.
    #. If ``a`` is a :math:`m \\times n` rectangular array of forms the
       forms in ``a`` are assembled into a matrix such that::

            A = [A_00 ... A_0n]
                [A_10 ... A_1n]
                [     ...     ]
                [A_m0 ..  A_mn]

       where ``A_ij`` is the matrix associated with the form
       ``a[i][j]``.

       a. If ``kind`` is a ``PETSc.Mat.Type`` (other than
          ``PETSc.Mat.Type.NEST``) or is ``None``, the matrix type is
          ``kind`` or the default type (if ``kind`` is ``None``).
       #. If ``kind`` is ``PETSc.Mat.Type.NEST`` or a rectangular array
          of PETSc matrix types, the returned matrix has type
          ``PETSc.Mat.Type.NEST``.

    Rows/columns that are constrained by a Dirichlet boundary condition
    are zeroed, with the diagonal to set to ``diag``.

    Constant and coefficient data that appear in the form(s) can be
    packed outside of this function to avoid re-packing by this
    function. The functions :func:`dolfinx.fem.pack_constants` and
    :func:`dolfinx.fem.pack_coefficients` can be used to 'pre-pack' the
    data.

    Note:
        The returned matrix is not 'assembled', i.e. ghost contributions
        are not accumulated.

    Args:
        a: Bilinear form(s) to assembled into a matrix.
        bcs: Dirichlet boundary conditions applied to the system.
        diag: Value to set on the matrix diagonal for Dirichlet
            boundary condition constrained degrees-of-freedom belonging
            to the same trial and test space.
        constants: Constants appearing in the form.
        coeffs: Coefficients appearing in the form.
        kind: PETSc matrix type (``MatType``).

    Returns:
        Matrix representing the bilinear form.

    Note:
        Convenience function for callers that have boundary conditions.
        It rebuilds the constrained dof markers on every call, and
        should not be called internally by the library.
    """  # noqa: D301
    return _assemble_matrix_mat(create_matrix(a, kind), a, bcs, diag, constants, coeffs)


@assemble_matrix.register  # type: ignore[attr-defined]
def _assemble_matrix_mat(
    A: PETSc.Mat,
    a: Form | Sequence[Sequence[Form | None]],
    bcs: Sequence[DirichletBC] | None = None,
    diag: float = 1,
    constants: npt.NDArray | Sequence[Sequence[npt.NDArray]] | None = None,
    coeffs: (
        dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]
        | Sequence[Sequence[dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]]]
        | None
    ) = None,
) -> PETSc.Mat:
    """Assemble bilinear form into a matrix.

    The matrix vector ``A`` must have been initialized with a
    size/layout that is consistent with the bilinear form(s). The PETSc
    matrix ``A`` is normally created by :func:`create_matrix`.

    The returned matrix is not finalised, i.e. ghost values are not
    accumulated.

    Note:
        Convenience function for callers that have boundary conditions.
        It rebuilds the constrained dof markers on every call, and
        should not be called internally by the library.
    """
    bc_data = _matrix_bc_data(a, bcs)
    return _assemble_matrix_petsc(
        A,
        a,
        bc_data,
        _matrix_diag_data(bc_data, diag, _diag_on_ghost_rows(A, a)),
        pack_constants(a) if constants is None else constants,
        pack_coefficients(a) if coeffs is None else coeffs,
    )


def _vector_kind(A: PETSc.Mat, kind, L) -> str | None:
    """PETSc vector type matching a matrix built with ``kind``.

    "nest" names both a matrix and a vector type, but MATIS has no Vec
    counterpart: a blocked problem needs the monolithic "mpi" layout
    that matches the matrix, and a single form the default type, since
    "mpi" would build a blocked vector and send the form down the
    blocked path.

    Args:
        A: Assembled matrix, whose type settles the nest case.
        kind: Matrix kind the problem was created with.
        L: Linear form(s), a sequence for a blocked problem.

    Returns:
        The vector type, or ``None`` for the default.
    """
    kind = "nest" if A.getType() == PETSc.Mat.Type.NEST else kind
    if kind == "is":
        kind = "mpi" if isinstance(L, Sequence) else None
    assert kind is None or isinstance(kind, str)
    return kind


def _field_dm(A: PETSc.Mat, u, forms) -> PETSc.DMShell:
    """DM carrying the field decomposition of a problem.

    Preconditioners such as PCBDDC and PCFIELDSPLIT use it to split the
    problem into fields. The caller attaches it to the preconditioner
    only: on the KSP it would have PETSc rebuild the operators via
    DMCreateMatrix.

    Args:
        A: Matrix the problem is assembled into.
        u: Solution function(s).
        forms: Form(s) giving the field layout.

    Returns:
        The DM, ready to attach.
    """
    dm = PETSc.DMShell().create(A.comm)
    dm.setCreateMatrix(functools.partial(_dm_create_matrix, A))  # type: ignore[missing-attribute]
    dm.setCreateFieldDecomposition(  # type: ignore[missing-attribute]
        functools.partial(_dm_create_field_decomposition, u, forms)
    )
    return dm


class _MatrixBCData(typing.NamedTuple):
    """Cached constrained dof markers, one entry per block row/column.

    The spaces the markers were built from are kept so that a repeated
    caller resolves them once: :func:`_block_index_sets` needs the same
    ones.
    """

    row_markers: list[npt.NDArray[np.int8]]
    column_markers: list[npt.NDArray[np.int8]]
    row_spaces: Sequence[_FunctionSpace | None]
    column_spaces: Sequence[_FunctionSpace | None]


def _extract_block_spaces(
    a: Sequence[Sequence[Form | None]],
) -> tuple[list[_FunctionSpace | None], list[_FunctionSpace | None]]:
    """Test and trial spaces of a 2D array of forms.

    Each block row and each block column must have its own function
    space, so that the block a dof belongs to, and the blocks a
    boundary condition on that space constrains, are unambiguous.

    Raises:
        ValueError: If one space is shared by two block rows or by two
            block columns.
    """
    row_spaces = _extract_function_spaces(a, 0)
    column_spaces = _extract_function_spaces(a, 1)
    for spaces, what in ((row_spaces, "rows"), (column_spaces, "columns")):
        for i, Vi in enumerate(spaces):
            if Vi is None:
                continue
            for j, Vj in enumerate(spaces[:i]):
                if Vj is not None and Vj._cpp_object is Vi._cpp_object:
                    raise ValueError(
                        f"Function space is shared by {what} {j} and {i} of a "
                        f"blocked form. Each of the {what} must have its own space. "
                        "Use FunctionSpace.clone() for separate blocks."
                    )
    return row_spaces, column_spaces


def _matrix_bc_data(
    a: Form | Sequence[Sequence[Form | None]], bcs: Sequence[DirichletBC] | None
) -> _MatrixBCData:
    """Reusable constrained dof markers and resolved spaces.

    Entries correspond to block rows and columns. A single form has one
    entry in each field. Markers for a shared test/trial space share an
    array, and column markers can also be used for lifting. Treat the
    arrays as read-only. Boundary condition values are not cached.
    """
    if isinstance(a, Sequence):
        V0, V1 = _extract_block_spaces(a)
    else:
        test_space, trial_space = a.function_spaces
        V0, V1 = [test_space], [trial_space]
    markers = _bc_dof_markers_by_space([*V0, *V1], bcs)
    row_markers, column_markers = markers[: len(V0)], markers[len(V0) :]
    return _MatrixBCData(row_markers, column_markers, V0, V1)


def _block_index_sets(bc_data: _MatrixBCData) -> tuple[list, list]:
    """Index sets addressing the block rows and columns of a form array.

    :meth:`PETSc.Mat.getLocalSubMatrix` takes these to reach a block of
    a blocked matrix. They follow from the function spaces, so a caller
    assembling repeatedly can build them once.

    Args:
        bc_data: Constrained dofs, from :func:`_matrix_bc_data`, whose
            resolved spaces give the layout.

    Returns:
        One index set per block row, and one per block column.

    Raises:
        ValueError: If a block row or column holds no form, leaving its
            space undetermined.
    """

    def sets(spaces, what):
        if all(V is None for V in spaces):
            raise ValueError(f"Cannot have an entire {what} of forms be 'None'.")
        return _cpp.la.petsc.create_index_sets(
            [
                (V.dofmaps[0].index_map._cpp_object, V.dofmaps[0].index_map_bs)  # type: ignore
                for V in spaces
            ]
        )

    return sets(bc_data.row_spaces, "row"), sets(bc_data.column_spaces, "column")


class _MatrixDiagData(typing.NamedTuple):
    """Rows carrying the constrained diagonal, and the value on each.

    One entry per block row, as :class:`_MatrixBCData`. A value is
    either a scalar for every row of its block, or one value per row.
    Both forms are accepted by :func:`dolfinx.la.petsc.set_diagonal`.
    """

    rows: list[npt.NDArray[np.int32]]
    values: list[npt.NDArray | float | complex]


def _check_nest_forms(A: PETSc.Mat, a: Form | Sequence[Sequence[Form | None]]) -> None:
    """Raise if a nest matrix is to be assembled from a single form.

    Args:
        A: Matrix to be assembled into.
        a: Bilinear form, or a 2D array of them.

    Raises:
        ValueError: If ``A`` is a nest and ``a`` is a single form.
    """
    if A.getType() == PETSc.Mat.Type.NEST and not isinstance(a, Sequence):
        raise ValueError("Must provide a sequence of forms when assembling a nest matrix")


def _diag_on_ghost_rows(A: PETSc.Mat, a: Form | Sequence[Sequence[Form | None]]) -> list[bool]:
    """Whether the diagonal is written on ghost rows as well as owned ones.

    An assembled matrix has its constrained rows written by the process
    owning them. An unassembled one (MATIS) has no insert stage in
    which an owner could claim a row, so every process holding a
    constrained row writes it, and a share of the value.

    A nest holds a matrix per block, which may differ in type: a MATIS
    velocity block beside an assembled pressure one, say. The diagonal
    of block row ``i`` goes into the sub-matrix ``(i, i)``, so that is
    the matrix governing it.

    Args:
        A: Matrix to be assembled into.
        a: Bilinear form, or a 2D array of them.

    Returns:
        One flag per block row.

    Raises:
        ValueError: If ``A`` is a nest and ``a`` is a single form.
    """
    _check_nest_forms(A, a)
    if A.getType() != PETSc.Mat.Type.NEST:
        n = len(a) if isinstance(a, Sequence) else 1
        return [A.getType() == PETSc.Mat.Type.IS] * n

    # A block row with no diagonal block, because it is 'None' or
    # because the nest is rectangular, carries no Dirichlet diagonal
    ncols = len(a[0]) if len(a) > 0 else 0  # type: ignore[arg-type,index]
    flags = []
    for i in range(len(a)):  # type: ignore[arg-type]
        sub = A.getNestSubMatrix(i, i) if i < ncols else None
        flags.append(
            False if sub is None or sub.handle == 0 else sub.getType() == PETSc.Mat.Type.IS
        )
    return flags


def _matis_diag_data(
    index_map: IndexMap,
    bs: int,
    dof_marker: npt.NDArray[np.int8],
    diagonal: float | complex,
) -> tuple[npt.NDArray[np.int32], npt.NDArray]:
    """Constrained rows, and this process's share of ``diagonal``.

    A MATIS matrix holds an unassembled local matrix per process, with
    no insert stage in which an owner could claim a row. Every
    constrained row is therefore written, ghosts included, and each
    process sharing a row writes ``diagonal`` divided by the number of
    processes sharing it. Summing the local matrices then recovers
    ``diagonal``, whatever the partition.

    Args:
        index_map: Parallel layout of the rows.
        bs: Block size relating the rows to the blocks of ``index_map``.
        dof_marker: Constrained dof markers, covering owned and ghost
            dofs of ``index_map``, unrolled by ``bs``.
        diagonal: Value the assembled diagonal is to take.

    Returns:
        The rows, and the value each is to be written with.

    Note:
        Collective, as the sharer counts are.
    """
    rows = np.flatnonzero(dof_marker).astype(np.int32)
    sharers = dolfinx.common.num_sharing_ranks(index_map, rows, bs)
    return rows, (diagonal / sharers).astype(PETSc.ScalarType)


def _matrix_diag_data(
    bc_data: _MatrixBCData,
    diagonal: float | complex,
    include_ghosts: Sequence[bool],
) -> _MatrixDiagData:
    """Resolve ``diagonal`` to the rows it is written on, and the values.

    Where only owned rows are written, the owned constrained rows that
    :func:`_matrix_bc_data` already found take ``diagonal`` as it
    stands. Where ghost rows are written too,
    :func:`_matis_diag_data` supplies both.

    The result depends only on the matrix type and on the dofs the
    boundary conditions constrain, both fixed, so a repeated caller
    should resolve once and reuse.

    Args:
        bc_data: Constrained dofs, from :func:`_matrix_bc_data`.
        diagonal: Value the assembled diagonal is to take.
        include_ghosts: Whether to write the diagonal on ghost rows as
            well as owned ones, one per block row, from
            :func:`_diag_on_ghost_rows`.

    Returns:
        The rows and values, one entry per block row.

    Note:
        Collective where a block row includes its ghosts.
        ``include_ghosts`` follows from the matrix, which every process
        sees alike, so all take the same branch.
    """
    rows: list[npt.NDArray[np.int32]] = []
    values: list[npt.NDArray | float | complex] = []
    for V, marker, ghost in zip(
        bc_data.row_spaces, bc_data.row_markers, include_ghosts, strict=True
    ):
        if V is None or not ghost:
            rows.append(np.empty(0, dtype=np.int32) if V is None else _owned_marked_rows(V, marker))
            values.append(diagonal)
        else:
            dofmap = V.dofmaps[0]
            row, value = _matis_diag_data(dofmap.index_map, dofmap.index_map_bs, marker, diagonal)
            rows.append(row)
            values.append(value)
    return _MatrixDiagData(rows, values)


def _assemble_matrix_single(
    A: PETSc.Mat,
    a: Form,
    dof_markers: tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]],
    rows: npt.NDArray[np.int32],
    diag: npt.NDArray | float | complex,
    constants: npt.NDArray,
    coeffs: dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray],
    *,
    unrolled: bool,
) -> None:
    """Assemble one form and set its constrained diagonal.

    ``rows`` and ``diag`` come from :func:`_matrix_diag_data`, which
    decides whether ghost rows are written and with what share.
    """
    _cpp.fem.petsc.assemble_matrix(
        A,
        a._cpp_object,  # type: ignore[arg-type]
        constants,
        coeffs,
        dof_markers,
        unrolled,
    )
    V0, V1 = a.function_spaces
    if V0._cpp_object is V1._cpp_object:
        # Adding to zeroed rows sets the diagonal without a flush.
        dolfinx.la.petsc.set_diagonal(A, rows, diag, PETSc.InsertMode.ADD)  # type: ignore[arg-type]


def _assemble_matrix_blocked(
    A: PETSc.Mat,
    a: Sequence[Sequence[Form | None]],
    bc_data: _MatrixBCData,
    diag_data: _MatrixDiagData,
    constants: Sequence[Sequence[npt.NDArray | None]],
    coeffs: Sequence[Sequence[dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]]],
    index_sets: tuple[Sequence, Sequence] | None,
) -> PETSc.Mat:
    """Assemble a form array block by block; do not zero or finalise ``A``.

    ``index_sets`` is ``None`` for a nest matrix, whose blocks are
    sub-matrices in their own right, and otherwise addresses the blocks
    of ``A`` as local sub-matrices, which are indexed by scalar rather
    than by block.
    """
    if index_sets is None:

        @contextlib.contextmanager
        def sub_matrix(i, j):
            yield A.getNestSubMatrix(i, j)

        unrolled = False
    else:
        is0, is1 = index_sets

        @contextlib.contextmanager
        def sub_matrix(i, j):
            Asub = A.getLocalSubMatrix(is0[i], is1[j])
            try:
                yield Asub
            finally:
                A.restoreLocalSubMatrix(is0[i], is1[j], Asub)

        unrolled = True

    for i, (a_row, const_row, coeff_row) in enumerate(zip(a, constants, coeffs, strict=True)):
        for j, (a_block, const, coeff) in enumerate(zip(a_row, const_row, coeff_row, strict=True)):
            if a_block is not None:
                assert const is not None
                with sub_matrix(i, j) as Asub:
                    _assemble_matrix_single(
                        Asub,
                        a_block,
                        (bc_data.row_markers[i], bc_data.column_markers[j]),
                        diag_data.rows[i],
                        diag_data.values[i],
                        const,
                        coeff,
                        unrolled=unrolled,
                    )
            elif i == j and bc_data.row_markers[i].size > 0:
                raise RuntimeError(
                    f"Diagonal sub-block ({i}, {j}) cannot be 'None'"
                    " and have DirichletBC applied. Consider assembling a zero block."
                )
    return A


def _assemble_matrix_petsc(
    A: PETSc.Mat,
    a: Form | Sequence[Sequence[Form | None]],
    bc_data: _MatrixBCData,
    diag_data: _MatrixDiagData,
    constants: npt.NDArray | Sequence[Sequence[npt.NDArray | None]],
    coeffs: (
        dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]
        | Sequence[Sequence[dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]]]
    ),
    index_sets: tuple[Sequence, Sequence] | None = None,
) -> PETSc.Mat:
    """Assemble form(s) with cached BC data; do not zero or finalise ``A``.

    ``bc_data`` supplies the row and column markers from
    :func:`_matrix_bc_data`, and ``diag_data`` the constrained rows and
    the value on each from :func:`_matrix_diag_data`. Both are fixed
    for the lifetime of the boundary conditions. Constants and
    coefficients are supplied afresh by the caller.
    """
    _check_nest_forms(A, a)
    if A.getType() == PETSc.Mat.Type.NEST:
        assert isinstance(a, Sequence)  # _check_nest_forms has ruled out a single form
        if isinstance(coeffs, dict):
            raise ValueError(
                "Must provide a sequence of sequences of coefficients when assembling a nest matrix"
            )
        return _assemble_matrix_blocked(
            A,
            a,
            bc_data,
            diag_data,
            constants,  # type: ignore[arg-type]
            coeffs,  # type: ignore[arg-type]
            None,
        )
    elif isinstance(a, Sequence):
        return _assemble_matrix_blocked(
            A,
            a,
            bc_data,
            diag_data,
            constants,  # type: ignore[arg-type]
            coeffs,  # type: ignore[arg-type]
            _block_index_sets(bc_data) if index_sets is None else index_sets,
        )
    else:
        _assemble_matrix_single(
            A,
            a,
            (bc_data.row_markers[0], bc_data.column_markers[0]),
            diag_data.rows[0],
            diag_data.values[0],
            constants,  # type: ignore[arg-type]
            coeffs,  # type: ignore[arg-type]
            unrolled=False,
        )
        return A


def apply_lifting(
    b: PETSc.Vec,
    a: Sequence[Form | None] | Sequence[Sequence[Form | None]],
    bcs: Sequence[DirichletBC] | Sequence[Sequence[DirichletBC]] | None,
    x0: Sequence[PETSc.Vec] | None = None,
    alpha: float = 1,
    constants: Sequence[npt.NDArray] | Sequence[Sequence[npt.NDArray]] | None = None,
    coeffs: (
        dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]
        | Sequence[Sequence[dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]]]
        | None
    ) = None,
) -> None:
    """Modify a vector to account for Dirichlet boundary conditions.

    See :func:`dolfinx.fem.apply_lifting` for a mathematical
    descriptions of the lifting operation.

    Args:
        b: Vector to modify in-place.
        a: List of bilinear forms. If ``b`` is not blocked or a nest,
            then ``a`` is a 1D sequence. If ``b`` is blocked or a nest,
            then ``a`` is  a 2D array of forms, with the ``a[i]`` forms
            used to modify the block/nest vector ``b[i]``.
        bcs: Boundary conditions to apply, which form a 2D array.
            If ``b`` is nested or blocked then ``bcs[i]`` are the
            boundary conditions to apply to block/nest ``i``.
            The function :func:`dolfinx.fem.bcs_by_block` can be
            used to prepare the 2D array of ``DirichletBC`` objects
            from the 2D sequence ``a``::

                bcs1 = fem.bcs_by_block(
                    fem.extract_function_spaces(a, 1),
                    bcs
                )

            If ``b`` is not blocked or nest, then ``len(bcs)`` must be
            equal to 1. The function :func:`dolfinx.fem.bcs_by_block`
            can be used to prepare the 2D array of ``DirichletBC``
            from the 1D sequence ``a``::

                bcs1 = fem.bcs_by_block(
                    fem.extract_function_spaces([a], 1),
                    bcs
                )

        x0: Vector to use in modify ``b`` (see
            :func:`dolfinx.fem.apply_lifting`). Treated as zero if
            ``None``.
        alpha: Scalar parameter in lifting (see
            :func:`dolfinx.fem.apply_lifting`).
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
    """
    _apply_lifting_petsc(
        b,
        a,
        _lifting_bc_markers(a, bcs),  # type: ignore[arg-type]
        _lifting_bc_values(a, bcs),  # type: ignore[arg-type]
        x0,
        alpha,
        constants,
        coeffs,  # type: ignore[arg-type]
    )


def _lifting_spaces(
    a: Sequence[Form | None] | Sequence[Sequence[Form | None]],
) -> list[_FunctionSpace | None]:
    """Trial space of each column of ``a``, ``None`` where it has no form.

    ``a`` is a 1D sequence with one form per column, or a 2D array of
    forms whose columns share a trial space.
    """
    if len(a) > 0 and isinstance(a[0], Sequence):
        return _extract_function_spaces(a, 1)  # type: ignore[arg-type,return-value]
    return [None if form is None else form.function_spaces[1] for form in a]  # type: ignore[union-attr]


def _lifting_bc_markers(
    a: Sequence[Form | None] | Sequence[Sequence[Form | None]],
    bcs: Sequence[Sequence[DirichletBC]] | None,
) -> list[npt.NDArray[np.int8]]:
    """Constrained dof markers for lifting, per column of ``a``.

    The ``bc_markers1`` argument of
    :func:`_apply_lifting_petsc`, with ``bcs[j]`` the
    conditions on column ``j`` (``None`` for none at all). Markers are
    fixed for the lifetime of ``bcs``, so a repeated caller should
    build them once and pair them with fresh values from
    :func:`_lifting_bc_values`.
    """
    spaces = _lifting_spaces(a)
    return _bc_lifting_markers(spaces, [[] for _ in spaces] if bcs is None else bcs)


def _lifting_bc_values(
    a: Sequence[Form | None] | Sequence[Sequence[Form | None]],
    bcs: Sequence[Sequence[DirichletBC]] | None,
) -> list[npt.NDArray]:
    """Boundary condition values for lifting, per column of ``a``.

    The ``bc_values1`` argument of
    :func:`_apply_lifting_petsc`, as ``PETSc.ScalarType``.
    Values are read from ``bcs`` on every call and must not be cached,
    since the function or constant behind a condition may have changed.
    """
    spaces = _lifting_spaces(a)
    return _bc_lifting_values(
        spaces, [[] for _ in spaces] if bcs is None else bcs, PETSc.ScalarType
    )


def _apply_lifting_petsc(
    b: PETSc.Vec,
    a: Sequence[Form | None] | Sequence[Sequence[Form | None]],
    bc_markers1: Sequence[npt.NDArray[np.int8]],
    bc_values1: Sequence[npt.NDArray],
    x0: Sequence[PETSc.Vec] | None = None,
    alpha: float = 1,
    constants: Sequence[npt.NDArray] | Sequence[Sequence[npt.NDArray]] | None = None,
    coeffs: (
        dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]
        | Sequence[Sequence[dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]]]
        | None
    ) = None,
) -> None:
    """Lifting (see :func:`apply_lifting`), given constrained dofs.

    ``bc_markers1[j]`` and ``bc_values1[j]`` are the constrained dof
    markers and boundary condition values on the trial space of column
    ``j``, from :func:`_lifting_bc_markers` and
    :func:`_lifting_bc_values`.
    """
    if b.getType() == PETSc.Vec.Type.NEST:
        x0 = [] if x0 is None else x0.getNestSubVecs()  # type: ignore[attr-defined]
        if constants is None:
            constants = [pack_constants(forms) for forms in a]  # type: ignore
        if coeffs is None:
            coeffs = [pack_coefficients(forms) for forms in a]  # type: ignore
        assert coeffs is not None
        assert constants is not None
        constants_ = typing.cast(Sequence[Sequence[npt.NDArray | None]], constants)
        coeffs_ = typing.cast(
            Sequence[Sequence[dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]]], coeffs
        )
        for b_sub, a_sub, const, coeff in zip(
            b.getNestSubVecs(),
            a,
            constants_,
            coeffs_,
            strict=True,
        ):
            const_ = [np.array([], dtype=PETSc.ScalarType) if x is None else x for x in const]
            _apply_lifting_petsc(
                b_sub,
                a_sub,  # type: ignore[arg-type]
                bc_markers1,
                bc_values1,
                x0,  # type: ignore[arg-type]
                alpha,
                const_,
                coeff,  # type: ignore[arg-type]
            )
    else:
        with contextlib.ExitStack() as stack:
            if b.getAttr("_blocks") is not None:
                if x0 is not None:
                    offset0, offset1 = x0.getAttr("_blocks")  # type: ignore[attr-defined]
                    xl = stack.enter_context(x0.localForm())  # type: ignore[attr-defined]
                    xl_r = xl.array_r
                    xlocal = [
                        np.concatenate((xl_r[off0:off1], xl_r[offg0:offg1]))
                        for (off0, off1, offg0, offg1) in zip(
                            offset0[:-1], offset0[1:], offset1[:-1], offset1[1:], strict=True
                        )
                    ]
                else:
                    xlocal = None

                offset0, offset1 = b.getAttr("_blocks")  # type: ignore
                with b.localForm() as b_l:
                    for i, (a_, off0, off1, offg0, offg1) in enumerate(
                        zip(a, offset0[:-1], offset0[1:], offset1[:-1], offset1[1:], strict=True)
                    ):
                        const = (
                            pack_constants(a_)
                            if constants is None
                            else typing.cast(Sequence[npt.NDArray | None], constants[i])
                        )
                        assert const is not None
                        coeff = (
                            pack_coefficients(a_)
                            if coeffs is None
                            else typing.cast(
                                Sequence[dict[tuple[dolfinx.fem.IntegralType, int], npt.NDArray]],
                                coeffs,
                            )[i]
                        )
                        const_ = [
                            np.empty(0, dtype=PETSc.ScalarType) if val is None else val
                            for val in const
                        ]
                        b_l_r = b_l.array_r
                        bx_ = np.concatenate((b_l_r[off0:off1], b_l_r[offg0:offg1]))
                        _apply_lifting_markers(
                            bx_,
                            a_,  # type: ignore[arg-type]
                            bc_markers1,
                            bc_values1,
                            xlocal,
                            float(alpha),
                            const_,
                            coeff,  # type: ignore[arg-type]
                        )
                        size = off1 - off0
                        b_l.array_w[off0:off1] = bx_[:size]
                        b_l.array_w[offg0:offg1] = bx_[size:]
            else:
                if x0 is None:
                    x0 = []
                x0 = [stack.enter_context(x.localForm()) for x in x0]
                x0_r = [x.array_r for x in x0]
                b_local = stack.enter_context(b.localForm())
                _apply_lifting_markers(
                    b_local.array_w,
                    a,  # type: ignore[arg-type]
                    bc_markers1,
                    bc_values1,
                    x0_r,
                    alpha,
                    constants,  # type: ignore[arg-type]
                    coeffs,  # type: ignore[arg-type]
                )


def set_bc(
    b: PETSc.Vec,
    bcs: Sequence[DirichletBC] | Sequence[Sequence[DirichletBC]],
    x0: PETSc.Vec | None = None,
    alpha: float = 1,
) -> None:
    """Set constraint (Dirchlet boundary condition) values in an vector.

    For degrees-of-freedoms that are constrained by a Dirichlet boundary
    condition, this function sets that degrees-of-freedom to ``alpha *
    (g - x0)``, where ``g`` is the boundary condition value.

    Only owned entries in ``b`` (owned by the MPI process) are modified
    by this function.

    Args:
        b: Vector to modify by setting  boundary condition values.
        bcs: Boundary conditions to apply. If ``b`` is nested or
            blocked, ``bcs`` is a 2D array and ``bcs[i]`` are the
            boundary conditions to apply to block/nest ``i``. Otherwise
            ``bcs`` should be a sequence of ``DirichletBC``. For
            block/nest problems, :func:`dolfinx.fem.bcs_by_block` can be
            used to prepare the 2D array of ``DirichletBC`` objects.
        x0: Vector used in the value that constrained entries are set
            to. If ``None``, ``x0`` is treated as zero.
        alpha: Scalar value used in the value that constrained entries
            are set to.
    """
    if len(bcs) == 0:
        return

    if not isinstance(bcs[0], Sequence):
        x0 = x0.array_r if x0 is not None else None  # type: ignore
        for bc in bcs:
            bc.set(b.array_w, x0, alpha)  # type: ignore
    elif b.getType() == PETSc.Vec.Type.NEST:
        _b = b.getNestSubVecs()
        x0 = len(_b) * [None] if x0 is None else x0.getNestSubVecs()  # type: ignore
        for b_sub, bc_block, x_sub in zip(_b, bcs, x0, strict=True):  # type: ignore[call-overload]
            if not isinstance(bc_block, Sequence):
                raise ValueError("Expected a sequence of DirichletBC for a nested vector.")
            set_bc(b_sub, bc_block, x_sub, alpha)
    else:  # block vector
        offset0, _ = b.getAttr("_blocks")  # type: ignore
        b_array = b.getArray(readonly=False)
        x_array = x0.getArray(readonly=True) if x0 is not None else None
        for bcs_block, off0, off1 in zip(bcs, offset0[:-1], offset0[1:], strict=True):
            x0_sub = x_array[off0:off1] if x0 is not None else None  # type: ignore[index]
            for bc in bcs_block:  # type: ignore[attr-defined]
                bc.set(b_array[off0:off1], x0_sub, alpha)


# -- High-level interface for KSP ---------------------------------------


_U = typing.TypeVar("_U", bound=_Function | Sequence[_Function])


# -- DMShell helpers ------------------------------------------------------


def _dm_create_field_decomposition(
    u: _Function | Sequence[_Function],
    form: Form | Sequence[Form],
    _dm: PETSc.DM,
):
    """Index sets for the fields of a problem, and their names.

    Preconditioners such as PCBDDC and PCFIELDSPLIT use these to split
    the problem into fields.

    Args:
        u: Function(s) tied to the solution vector.
        form: Form of the residual or of the right-hand side. It can be
            a sequence of forms.
        _dm: The DM instance.

    Returns:
        Field names, index sets in global numbering, and sub-DMs. No
        sub-DMs are provided, so ``None`` is returned for them.
    """
    forms = form if isinstance(form, Sequence) else [form]
    spaces = _extract_function_spaces(forms)
    ises = _cpp.la.petsc.create_global_index_sets(
        [
            (V.dofmaps[0].index_map._cpp_object, V.dofmaps[0].index_map_bs)  # type: ignore[union-attr]
            for V in spaces
        ]
    )
    if isinstance(u, Sequence):
        # These become PETSc option prefixes, so an unnamed Function
        # (the default name is "f") contributes only its index, giving
        # the conventional "fieldsplit_0_" rather than "fieldsplit_f_0_"
        names = [f"{v.name + '_' if v.name != 'f' else ''}{i}" for i, v in enumerate(u)]
    else:
        names = [f"dolfinx_field_{i}" for i in range(len(forms))]
    return names, ises, None


def _dm_create_matrix(J: PETSc.Mat, _dm: PETSc.DM):
    """Duplicate the matrix layout.

    Args:
        J: Matrix to duplicate.
        _dm: The DM instance.
    """
    return J.duplicate()


class LinearProblem(typing.Generic[_U]):
    """High-level class for solving linears problem using a PETSc KSP.

    Solves problems of the form
    :math:`a_{ij}(u, v) = f_i(v), i,j=0,\\ldots,N\\
    \\forall v \\in V` where
    :math:`u=(u_0,\\ldots,u_N), v=(v_0,\\ldots,v_N)`
    using PETSc KSP as the linear solver.

    Note:
        This high-level class automatically handles PETSc memory
        management. The user does not need to manually call
        ``.destroy()`` on returned PETSc objects.
    """  # noqa: D301

    @overload
    def __init__(
        self: LinearProblem[_Function],
        a: ufl.Form,
        L: ufl.Form,
        *,
        petsc_options_prefix: str,
        bcs: Sequence[DirichletBC] | None = None,
        u: _Function | None = None,
        P: ufl.Form | None = None,
        kind: str | None = None,
        petsc_options: dict | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: Sequence[_EntityMap] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: LinearProblem[Sequence[_Function]],
        a: Sequence[Sequence[ufl.Form | None]],
        L: Sequence[ufl.Form],
        *,
        petsc_options_prefix: str,
        bcs: Sequence[DirichletBC] | None = None,
        u: Sequence[_Function] | None = None,
        P: Sequence[Sequence[ufl.Form | None]] | None = None,
        kind: str | Sequence[Sequence[str | None]] | None = None,
        petsc_options: dict | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: Sequence[_EntityMap] | None = None,
    ) -> None: ...
    def __init__(
        self,
        a: ufl.Form | Sequence[Sequence[ufl.Form | None]],
        L: ufl.Form | Sequence[ufl.Form],
        *,
        petsc_options_prefix: str,
        bcs: Sequence[DirichletBC] | None = None,
        u: _Function | Sequence[_Function] | None = None,
        P: ufl.Form | Sequence[Sequence[ufl.Form | None]] | None = None,
        kind: str | Sequence[Sequence[str | None]] | None = None,
        petsc_options: dict | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: Sequence[_EntityMap] | None = None,
    ) -> None:
        """Initialize solver for a linear variational problem.

        By default, the underlying KSP solver uses PETSc's default
        options, usually GMRES + ILU preconditioning. To use the robust
        combination of LU via MUMPS

        Example::

            problem = LinearProblem(a, L, bcs=[bc0, bc1],
                petsc_options_prefix="basic_linear_problem",
                petsc_options= {
                  "ksp_type": "preonly",
                  "pc_type": "lu",
                  "pc_factor_mat_solver_type": "mumps"
            })

        This class also supports nested block-structured problems.

        Example::

            problem = LinearProblem([[a00, a01], [None, a11]], [L0, L1],
                bcs=[bc0, bc1], u=[uh0, uh1],
                kind="nest",
                petsc_options_prefix="nest_linear_problem")

        Every PETSc object created will have a unique options prefix set.
        We recommend discovering these prefixes dynamically via the
        petsc4py API rather than hard-coding each prefix value
        into the programme.

        Example::

            ksp_options_prefix = problem.solver.getOptionsPrefix()
            A_options_prefix = problem.A.getOptionsPrefix()

        Args:
            a: Bilinear UFL form or a nested sequence of bilinear
                forms, the left-hand side of the variational problem.
            L: Linear UFL form or a sequence of linear forms, the
                right-hand side of the variational problem.
            bcs: Sequence of Dirichlet boundary conditions to apply to
                 the variational problem and the preconditioner matrix.
            u: Solution function. It is created if not provided.
            P: Bilinear UFL form or a sequence of sequence of bilinear
                forms, used as a preconditioner. Must be over the same
                function spaces as ``a``.
            kind: The PETSc matrix and vector kind. Common choices
                are ``mpi`` and ``nest``. See
                :func:`dolfinx.fem.petsc.create_matrix` and
                :func:`dolfinx.fem.petsc.create_vector` for more
                information.
            petsc_options_prefix: Mandatory named argument. Options prefix
                used as root prefix on all internally created PETSc
                objects. Typically ends with ``_``. Must be the same on
                all ranks, and is usually unique within the programme.
            petsc_options: Options set on the underlying PETSc KSP only.
                The options must be the same on all ranks. For available
                choices for the ``petsc_options`` kwarg, see the `PETSc KSP
                documentation
                <https://petsc4py.readthedocs.io/en/stable/manual/ksp/>`_.
                Options on other objects (matrices, vectors) should be set
                explicitly by the user.
            form_compiler_options: Options used in FFCx compilation of
                all forms. Run ``ffcx --help`` at the commandline to see
                all available options.
            jit_options: Options used in CFFI JIT compilation of C
                code generated by FFCx. See ``python/dolfinx/jit.py`` for
                all available options. Takes priority over all other
                option values.
            entity_maps: If any trial functions, test functions, or
                coefficients in the form are not defined over the same mesh
                as the integration domain, a corresponding
                :class:`EntityMap <dolfinx.mesh.EntityMap>`
                must be provided.
        """
        self._a = _create_form(
            a,
            dtype=PETSc.ScalarType,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )
        self._L = _create_form(
            L,
            dtype=PETSc.ScalarType,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )
        self._A = create_matrix(self._a, kind=kind)
        self._preconditioner = _create_form(
            P,
            dtype=PETSc.ScalarType,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )
        _check_preconditioner_spaces(self._a, self._preconditioner)
        self._P_mat = (
            create_matrix(self._preconditioner, kind=kind)
            if self._preconditioner is not None
            else None
        )

        kind = _vector_kind(self.A, kind, self.L)
        self._b = _create_vector_from_form(self.L, kind=kind)
        self._x = _create_vector_from_form(self.L, kind=kind)

        self._u: _Function | Sequence[_Function]
        if u is None:
            # Extract function space for unknown from the right hand
            # side of the equation.
            if isinstance(L, Sequence):
                self._u = [_Function(Li.arguments()[0].ufl_function_space()) for Li in L]
            else:
                self._u = _Function(L.arguments()[0].ufl_function_space())
        else:
            self._u = u

        self.bcs = bcs

        self._solver = PETSc.KSP().create(self.A.comm)
        self.solver.setOperators(self.A, self.P_mat)

        self.solver.getPC().setDM(_field_dm(self.A, self._u, self.L))

        if petsc_options_prefix == "":
            raise ValueError("PETSc options prefix cannot be empty.")

        self._petsc_options_prefix = petsc_options_prefix
        self.solver.setOptionsPrefix(petsc_options_prefix)
        self.A.setOptionsPrefix(f"{petsc_options_prefix}A_")
        self.b.setOptionsPrefix(f"{petsc_options_prefix}b_")
        self.x.setOptionsPrefix(f"{petsc_options_prefix}x_")
        if self.P_mat is not None:
            self.P_mat.setOptionsPrefix(f"{petsc_options_prefix}P_mat_")

        # Set options on KSP only
        if petsc_options is not None:
            opts = PETSc.Options()
            opts.prefixPush(self.solver.getOptionsPrefix())

            for k, v in petsc_options.items():
                opts[k] = v  # type: ignore

            self.solver.setFromOptions()

            # Tidy up global options
            for k in petsc_options.keys():
                del opts[k]  # type: ignore

            opts.prefixPop()

        if kind == "nest":
            # Transfer nest IS on self.A to PC of main KSP. This allows
            # fieldsplit preconditioning to be applied, if desired.
            nest_IS = self.A.getNestISs()
            fieldsplit_IS = tuple(
                [
                    (f"{u.name + '_' if u.name != 'f' else ''}{i}", IS)
                    for i, (u, IS) in enumerate(zip(self.u, nest_IS[0], strict=True))
                ]
            )
            self.solver.getPC().setFieldSplitIS(*fieldsplit_IS)

    def __del__(self) -> None:
        """Destroy internally held PETSc objects."""
        # __init__ may have raised before all attributes were set
        for name in ("_solver", "_A", "_b", "_x", "_P_mat"):
            if (obj := getattr(self, name, None)) is not None:
                obj.destroy()

    def solve(self) -> _U:
        """Solve the problem.

        This method updates the solution ``u`` function(s) stored in the
        problem instance.

        Note:
            The user is responsible for asserting convergence of the KSP
            solver e.g. ``problem.solver.getConvergedReason() > 0``.
            Alternatively, pass ``"ksp_error_if_not_converged" : True`` in
            ``petsc_options`` to raise a ``PETScError`` on failure.

        Returns:
            The solution function(s).
        """
        # Assemble lhs
        self.A.zeroEntries()
        _assemble_matrix_petsc(
            self.A,
            self.a,
            self._a_bc_data,
            self._a_diag_data,
            pack_constants(self.a),
            pack_coefficients(self.a),
        )
        self.A.assemble()

        # Assemble preconditioner
        if self.preconditioner is not None:
            assert self.P_mat is not None
            self.P_mat.zeroEntries()
            # Built from the same 'kind' as A and forced onto the same
            # spaces, so the preconditioner constrains the same dofs and
            # writes the same diagonal
            _assemble_matrix_petsc(
                self.P_mat,
                self.preconditioner,
                self._a_bc_data,
                self._a_diag_data,
                pack_constants(self.preconditioner),
                pack_coefficients(self.preconditioner),
            )
            self.P_mat.assemble()

        # Assemble rhs
        dolfinx.la.petsc._zero_vector(self.b)
        _assemble_vector_petsc(self.b, self.L)

        # Apply boundary conditions to the rhs
        a, L = self.a, self.L
        block = isinstance(self.u, Sequence)  # block or nest
        if block and not (isinstance(a, Sequence) and isinstance(L, Sequence)):
            raise ValueError("Expected a sequence of forms for a block/nest problem.")
        elif not block and isinstance(a, Sequence):
            raise ValueError("Expected a single form for a non-block/nest problem.")
        _apply_lifting_petsc(
            self.b,
            a if block else [a],  # type: ignore[arg-type]
            self._a_bc_data.column_markers,
            _bc_lifting_values(self._a_bc_data.column_spaces, self._bcs1, PETSc.ScalarType),
        )
        dolfinx.la.petsc._ghost_update(
            self.b,
            PETSc.InsertMode.ADD,  # type: ignore[arg-type]
            PETSc.ScatterMode.REVERSE,  # type: ignore[arg-type]
        )
        if block:
            assert self._bcs0 is not None
            dolfinx.fem.petsc.set_bc(self.b, self._bcs0)
        else:
            for bc in self.bcs:
                bc.set(self.b.array_w)
        # Solve linear system and update ghost values in the solution
        self.solver.solve(self.b, self.x)
        dolfinx.la.petsc._ghost_update(self.x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore[arg-type]
        dolfinx.fem.petsc.assign(self.x, self.u)  # type: ignore
        return self.u

    @property
    def bcs(self) -> tuple[DirichletBC, ...]:
        """Dirichlet boundary conditions applied to the problem.

        Assigning rebuilds the cached dof markers, diagonal rows and
        per-block grouping that :meth:`solve` reuses. These follow from
        the dofs a condition constrains, which are fixed once it is
        built, and the sequence is copied, so the cache cannot go
        stale. Condition *values* are re-read on every solve.
        """
        return self._bcs

    @bcs.setter
    def bcs(self, bcs: Sequence[DirichletBC] | None) -> None:
        self._bcs = tuple(bcs) if bcs is not None else ()
        self._a_bc_data = _matrix_bc_data(self.a, self._bcs)
        # Resolving the diagonal is collective for a MATIS operator, so
        # it is done here rather than on every solve
        self._a_diag_data = _matrix_diag_data(
            self._a_bc_data, 1, _diag_on_ghost_rows(self.A, self.a)
        )
        # Which block each condition belongs to follows from the spaces,
        # so group once here and re-read only the values in solve().
        L = self.L
        if isinstance(self.u, Sequence):  # block or nest
            self._bcs1 = _bcs_by_block(self._a_bc_data.column_spaces, self._bcs)
            # A single form here is rejected by solve(), which reads these.
            self._bcs0 = (
                _bcs_by_block(_extract_function_spaces(L), self._bcs)
                if isinstance(L, Sequence)
                else None
            )
        else:  # single form, one column holding every condition
            self._bcs1 = [list(self._bcs)]
            self._bcs0 = None

    @property
    def L(self) -> Form | Sequence[Form]:
        """The compiled linear form representing the left-hand side."""
        return typing.cast(Form | Sequence[Form], self._L)

    @property
    def a(self) -> Form | Sequence[Sequence[Form]]:
        """The compiled bilinear form representing the right-hand side."""
        return typing.cast(Form | Sequence[Sequence[Form]], self._a)

    @property
    def preconditioner(self) -> Form | Sequence[Sequence[Form | None]] | None:
        """The compiled bilinear form representing the preconditioner."""
        return self._preconditioner

    @property
    def A(self) -> PETSc.Mat:
        """Left-hand side matrix."""
        return self._A

    @property
    def P_mat(self) -> PETSc.Mat | None:
        """Preconditioner matrix."""
        return self._P_mat

    @property
    def b(self) -> PETSc.Vec:
        """Right-hand side vector."""
        return self._b

    @property
    def x(self) -> PETSc.Vec:
        """Solution vector.

        Note:
            The vector does not share memory with the solution
            function(s) ``u``.
        """
        return self._x

    @property
    def solver(self) -> PETSc.KSP:
        """The PETSc KSP solver."""
        return self._solver

    @property
    def u(self) -> _U:
        """Solution function(s).

        Note:
            The function(s) do not share memory with the solution
            vector ``x``.
        """
        return self._u  # type: ignore[return-value]


# -- High-level interface for SNES ---------------------------------------


class _ResidualAssemblyData(typing.NamedTuple):
    """Fixed BC grouping and markers, with reusable lifting storage."""

    markers: Sequence[npt.NDArray[np.int8]]
    values: list[npt.NDArray]
    trial_bcs: Sequence[Sequence[DirichletBC]]
    residual_bcs: Sequence[DirichletBC] | Sequence[Sequence[DirichletBC]]


class _JacobianAssemblyData(typing.NamedTuple):
    """Fixed constrained rows, diagonal values and block addressing."""

    bc_data: _MatrixBCData
    diagonal: _MatrixDiagData
    preconditioner_diagonal: _MatrixDiagData
    index_sets: tuple[Sequence, Sequence] | None


def assemble_residual(
    _snes: PETSc.SNES,
    x: PETSc.Vec,
    b: PETSc.Vec,
    u: _Function | Sequence[_Function],
    residual: Form | Sequence[Form],
    jacobian: Form | Sequence[Sequence[Form]],
    bcs: Sequence[DirichletBC],
    _blocks: tuple[tuple[int, int, int], ...] | None = None,
    *,
    _assembly_data: _ResidualAssemblyData | None = None,
) -> None:
    """Assemble the residual at ``x`` into the vector ``b``.

    A function conforming to the interface expected by ``SNES.setFunction``
    by setting all arguments except `snes`, `x` and `b` through the `kargs`
    keyword argument.

    Example::

        snes = PETSc.SNES().create(mesh.comm)
        cntx = {"u": u, "residual": residual, "jacobian": jacobian,
            "bcs": bcs}
        snes.setFunction(assemble_residual, b, kargs=cntx)

    Note:
        The ``b`` passed in is not always the vector given to
        ``SNES.setFunction``: a line search, for instance, evaluates the
        residual in a work vector duplicated from it. Always assemble into
        the ``b`` this function receives, not a vector cached elsewhere.

    Args:
        _snes: The solver instance.
        x: The vector containing the point to evaluate the residual at.
        b: Vector to assemble the residual into.
        u: Function(s) tied to the solution vector within the residual and
           Jacobian.
        residual: Form of the residual. It can be a sequence of forms.
        jacobian: Form of the Jacobian. It can be a nested sequence of
            forms.
        bcs: List of Dirichlet boundary conditions to lift the residual.
        _blocks: If block assembly is requested this should contain the
            ownership layout for each block.
            See :func:`dolfinx.fem.petsc.create_vector` for more details
            on the format of this argument.

        _assembly_data: Internal BC data prepared by ``NonlinearProblem``.
            Must match the forms and boundary conditions.

    Note:
        The lifting markers are rebuilt from ``bcs`` on every call.
        :class:`NonlinearProblem` builds them once and reuses them,
        which it can do because the dofs a condition constrains are
        fixed when it is built.
    """
    _assemble_residual(_snes, x, b, u, residual, jacobian, bcs, _blocks, _assembly_data)


def _assemble_residual(
    _snes: PETSc.SNES,
    x: PETSc.Vec,
    b: PETSc.Vec,
    u: _Function | Sequence[_Function],
    residual: Form | Sequence[Form],
    jacobian: Form | Sequence[Sequence[Form]],
    bcs: Sequence[DirichletBC],
    _blocks: tuple[tuple[int, int, int], ...] | None,
    assembly_data: _ResidualAssemblyData | None,
) -> None:
    """Assemble with fixed BC data and current boundary values."""
    # Update input vector before assigning
    dolfinx.la.petsc._ghost_update(x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore[arg-type]

    # Assign the input vector to the unknowns
    assign(x, u)  # type: ignore

    # Assign block data if block assembly is requested
    if isinstance(residual, Sequence) and b.getType() != PETSc.Vec.Type.NEST:
        if _blocks is None:
            raise ValueError("Block data must be provided for block assembly.")
        b.setAttr("_blocks", _blocks)
        x.setAttr("_blocks", _blocks)

    # Assemble the residual
    dolfinx.la.petsc._zero_vector(b)
    _assemble_vector_petsc(b, residual)

    if assembly_data is not None:
        for values, conditions in zip(assembly_data.values, assembly_data.trial_bcs, strict=True):
            for bc in conditions:
                bc.set(values, None, 1)

    # Lift vector
    if isinstance(jacobian, Sequence):
        # Nest and blocked lifting
        if not isinstance(residual, Sequence):
            raise ValueError("Expected a sequence of forms for a block/nest residual.")
        if assembly_data is None:
            bcs1 = _bcs_by_block(_extract_function_spaces(jacobian, 1), bcs)
            markers = _lifting_bc_markers(jacobian, bcs1)
            values = _lifting_bc_values(jacobian, bcs1)
            bcs0 = _bcs_by_block(_extract_function_spaces(residual), bcs)
        else:
            markers, values = assembly_data.markers, assembly_data.values
            bcs0 = assembly_data.residual_bcs
        _apply_lifting_petsc(
            b,
            jacobian,
            markers,
            values,
            x0=x,  # type: ignore[arg-type]
            alpha=-1.0,
        )
        dolfinx.la.petsc._ghost_update(b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore[arg-type]
        set_bc(b, bcs0, x0=x, alpha=-1.0)  # type: ignore[arg-type]
    else:
        # Single form lifting
        if assembly_data is None:
            markers = _lifting_bc_markers([jacobian], [bcs])
            values = _lifting_bc_values([jacobian], [bcs])
        else:
            markers, values = assembly_data.markers, assembly_data.values
        _apply_lifting_petsc(
            b,
            [jacobian],
            markers,
            values,
            x0=[x],
            alpha=-1.0,
        )
        dolfinx.la.petsc._ghost_update(b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore[arg-type]
        set_bc(b, bcs, x0=x, alpha=-1.0)
    dolfinx.la.petsc._ghost_update(b, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore[arg-type]


def _check_preconditioner_spaces(
    a: Form | Sequence[Sequence[Form | None]],
    preconditioner: Form | Sequence[Sequence[Form | None]] | None,
) -> None:
    """Check that the preconditioner is over the operator's spaces.

    The preconditioner is assembled with the operator's constrained dof
    markers, so the two must be over the same function space objects,
    block for block. Equivalent spaces built separately are rejected.

    Args:
        a: Form(s) of the operator, i.e. the left-hand side of a linear
            problem or the Jacobian of a nonlinear one.
        preconditioner: Form(s) of the preconditioner, or ``None``.

    Raises:
        ValueError: If the shapes or the spaces differ.
    """
    if preconditioner is None:
        return
    message = (
        "Preconditioner form must be over the same function space objects as the "
        "operator it preconditions, not separately built equivalents."
    )
    if isinstance(a, Sequence):
        if not isinstance(preconditioner, Sequence):
            raise ValueError(message)
        spaces = [
            (_extract_function_spaces(a, i), _extract_function_spaces(preconditioner, i))
            for i in range(2)
        ]
    else:
        if isinstance(preconditioner, Sequence):
            raise ValueError(message)
        spaces = [(a.function_spaces, preconditioner.function_spaces)]
    for A_spaces, P_spaces in spaces:
        if len(A_spaces) != len(P_spaces):
            raise ValueError(message)
        for VA, VP in zip(A_spaces, P_spaces, strict=True):
            if VA is None or VP is None:
                if VA is not VP:
                    raise ValueError(message)
            elif VA._cpp_object is not VP._cpp_object:
                raise ValueError(message)


def assemble_jacobian(
    _snes: PETSc.SNES,
    x: PETSc.Vec,
    J: PETSc.Mat,
    P_mat: PETSc.Mat,
    u: Sequence[_Function] | _Function,
    jacobian: Form | Sequence[Sequence[Form]],
    preconditioner: Form | Sequence[Sequence[Form | None]] | None,
    bcs: Sequence[DirichletBC],
    *,
    _assembly_data: _JacobianAssemblyData | None = None,
) -> None:
    """Assemble the Jacobian and preconditioner matrices.

    A function conforming to the interface expected by
    ``SNES.setJacobian`` can be created by setting all arguments
    except `_snes`, `x`, `J` and `P_mat` through the `kargs` argument
    e.g.:

    Example::

        snes = PETSc.SNES().create(mesh.comm)
        cntx = {"u": u, "jacobian": jacobian,
            "preconditioner": preconditioner, "bcs": bcs}
        snes.setJacobian(assemble_jacobian, A, P_mat, kargs=cntx)

    Note:
        The ``J`` and ``P_mat`` passed in are not always the matrices given
        to ``SNES.setJacobian``. Always assemble into the matrices this
        function receives, not ones cached elsewhere.

    Args:
        _snes: The solver instance.
        x: Vector containing the point to evaluate at.
        J: Matrix to assemble the Jacobian into.
        P_mat: Matrix to assemble the preconditioner into.
        u: Function tied to the solution vector within the residual and
            Jacobian.
        jacobian: Compiled form of the Jacobian.
        preconditioner: Compiled form of the preconditioner, over the
            function spaces of ``jacobian``.
        bcs: Dirichlet boundary conditions to apply to the Jacobian and
            preconditioner matrices.
        _assembly_data: Internal BC data prepared by ``NonlinearProblem``.
            Must match the forms, boundary conditions and matrix types.
    """
    if _assembly_data is None:
        _check_preconditioner_spaces(jacobian, preconditioner)
        bc_data = _matrix_bc_data(jacobian, bcs)
        diag_data = _matrix_diag_data(bc_data, 1.0, _diag_on_ghost_rows(J, jacobian))
        preconditioner_diag = (
            _matrix_diag_data(bc_data, 1.0, _diag_on_ghost_rows(P_mat, preconditioner))
            if preconditioner is not None and P_mat != J
            else diag_data
        )
        index_sets = None
    else:
        bc_data = _assembly_data.bc_data
        diag_data = _assembly_data.diagonal
        preconditioner_diag = _assembly_data.preconditioner_diagonal
        index_sets = _assembly_data.index_sets

    # Copy existing solution into the function used in the residual and
    # Jacobian
    dolfinx.la.petsc._ghost_update(x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore[arg-type]
    assign(x, u)  # type: ignore

    # Assemble Jacobian
    J.zeroEntries()
    _assemble_matrix_petsc(
        J,
        jacobian,
        bc_data,
        diag_data,
        pack_constants(jacobian),
        pack_coefficients(jacobian),
        index_sets,
    )
    J.assemble()
    if preconditioner is not None:
        # Assembled with the same markers, rows and values, which
        # requires it to be over the spaces of the Jacobian and into a
        # matrix of the same type
        P_mat.zeroEntries()
        _assemble_matrix_petsc(
            P_mat,
            preconditioner,
            bc_data,
            preconditioner_diag,
            pack_constants(preconditioner),
            pack_coefficients(preconditioner),
            index_sets,
        )
        P_mat.assemble()


class NonlinearProblem(typing.Generic[_U]):
    """High-level class for solving nonlinear problems with PETSc SNES.

    Solves problems of the form
    :math:`F_i(u, v) = 0, i=0,\\ldots,N\\ \\forall v \\in V` where
    :math:`u=(u_0,\\ldots,u_N), v=(v_0,\\ldots,v_N)` using PETSc
    SNES as the non-linear solver.

    Note:
        This high-level class automatically handles PETSc memory
        management. The user does not need to manually call
        ``.destroy()`` on returned PETSc objects.
    """  # noqa: D301

    _P_mat: PETSc.Mat | None
    _preconditioner: Form | Sequence[Sequence[Form | None]] | None

    @overload
    def __init__(
        self: NonlinearProblem[_Function],
        F: ufl.form.Form,
        u: _Function,
        *,
        petsc_options_prefix: str,
        bcs: Sequence[DirichletBC] | None = None,
        J: ufl.form.Form | None = None,
        P: ufl.form.Form | None = None,
        kind: str | None = None,
        petsc_options: dict | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: Sequence[_EntityMap] | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: NonlinearProblem[Sequence[_Function]],
        F: Sequence[ufl.form.Form],
        u: Sequence[_Function],
        *,
        petsc_options_prefix: str,
        bcs: Sequence[DirichletBC] | None = None,
        J: Sequence[Sequence[ufl.form.Form]] | None = None,
        P: Sequence[Sequence[ufl.form.Form]] | None = None,
        kind: str | Sequence[Sequence[str | None]] | None = None,
        petsc_options: dict | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: Sequence[_EntityMap] | None = None,
    ) -> None: ...
    def __init__(
        self,
        F: ufl.form.Form | Sequence[ufl.form.Form],
        u: _Function | Sequence[_Function],
        *,
        petsc_options_prefix: str,
        bcs: Sequence[DirichletBC] | None = None,
        J: ufl.form.Form | Sequence[Sequence[ufl.form.Form]] | None = None,
        P: ufl.form.Form | Sequence[Sequence[ufl.form.Form]] | None = None,
        kind: str | Sequence[Sequence[str | None]] | None = None,
        petsc_options: dict | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: Sequence[_EntityMap] | None = None,
    ):
        """Initialize solver for a nonlinear variational problem.

        By default, the underlying SNES solver uses PETSc's default
        options. To use the robust combination of LU via MUMPS with
        a backtracking linesearch, pass:

        Example::

            petsc_options = {"ksp_type": "preonly",
                             "pc_type": "lu",
                             "pc_factor_mat_solver_type": "mumps",
                             "snes_linesearch_type": "bt",
            }

        Every PETSc object will have a unique options prefix set. We
        recommend discovering these prefixes dynamically via the
        petsc4py API rather than hard-coding each prefix value into
        the programme.

        Example::

            snes_options_prefix = problem.solver.getOptionsPrefix()
            jacobian_options_prefix = problem.A.getOptionsPrefix()

        Args:
            F: UFL form(s) representing the residual :math:`F_i`.
            u: Function(s) used to define the residual and Jacobian.
            bcs: Dirichlet boundary conditions, copied into an immutable
                tuple. Assign to :attr:`bcs` to change the conditions.
            J: UFL form(s) representing the Jacobian
                :math:`J_{ij} = dF_i/du_j`. If not passed, derived
                automatically.
            P: UFL form(s) representing the preconditioner, over the
                same function spaces as the Jacobian.
            kind: The PETSc matrix and vector kind. Common choices
                are ``mpi`` and ``nest``. See
                :func:`dolfinx.fem.petsc.create_matrix` and
                :func:`dolfinx.fem.petsc.create_vector` for more
                information.
            petsc_options_prefix: Mandatory named argument.
                Options prefix used as root prefix on all
                internally created PETSc objects. Typically ends with `_`.
                Must be the same on all ranks, and is usually unique within
                the programme.
            petsc_options: Options set on the underlying PETSc SNES only.
                The options must be the same on all ranks. For available
                choices for ``petsc_options``, see the
                `PETSc SNES documentation
                <https://petsc4py.readthedocs.io/en/stable/manual/snes/>`_.
                Options on other objects (matrices, vectors) should be set
                explicitly by the user.
            form_compiler_options: Options used in FFCx compilation of all
                forms. Run ``ffcx --help`` at the command line to see all
                available options.
            jit_options: Options used in CFFI JIT compilation of C code
                generated by FFCx. See ``python/dolfinx/jit.py`` for all
                available options. Takes priority over all other option
                values.
            entity_maps: If any trial functions, test functions, or
                coefficients in the form are not defined over the same mesh
                as the integration domain, a corresponding
                :class:`EntityMap <dolfinx.mesh.EntityMap>`
                must be provided.
        """
        # Compile residual and Jacobian forms
        self._F = _create_form(
            F,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )

        if J is None:
            J = typing.cast(typing.Any, derivative_block)(F, u)

        self._J = _create_form(
            J,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )

        if P is not None:
            self._preconditioner = _create_form(
                P,
                form_compiler_options=form_compiler_options,
                jit_options=jit_options,
                entity_maps=entity_maps,
            )
        else:
            self._preconditioner = None

        _check_preconditioner_spaces(self.J, self.preconditioner)
        self._u = u

        # Create PETSc structures for the residual, Jacobian and solution
        # vector
        self._A = create_matrix(self.J, kind=kind)
        # Create PETSc structure for preconditioner if provided
        if self._preconditioner is not None:
            self._P_mat = create_matrix(self._preconditioner, kind=kind)
        else:
            self._P_mat = None

        kind = _vector_kind(self._A, kind, self.F)
        self._b = _create_vector_from_form(self.F, kind=kind)
        self._x = _create_vector_from_form(self.F, kind=kind)

        # Create the SNES solver and attach the corresponding Jacobian and
        # residual computation functions
        self._snes = PETSc.SNES().create(self.A.comm)

        self.solver.getKSP().getPC().setDM(_field_dm(self.A, self._u, self.F))

        self.bcs = bcs

        if petsc_options_prefix == "":
            raise ValueError("PETSc options prefix cannot be empty.")

        self.solver.setOptionsPrefix(petsc_options_prefix)
        self.A.setOptionsPrefix(f"{petsc_options_prefix}A_")
        if self.P_mat is not None:
            self.P_mat.setOptionsPrefix(f"{petsc_options_prefix}P_mat_")
        self.b.setOptionsPrefix(f"{petsc_options_prefix}b_")
        self.x.setOptionsPrefix(f"{petsc_options_prefix}x_")

        # Set options for SNES only
        if petsc_options is not None:
            opts = PETSc.Options()
            opts.prefixPush(self.solver.getOptionsPrefix())

            for k, v in petsc_options.items():
                opts[k] = v  # type: ignore

            self.solver.setFromOptions()

            # Tidy up global options
            for k in petsc_options.keys():
                del opts[k]  # type: ignore

            opts.prefixPop()

        if self.P_mat is not None and kind == "nest":
            # Transfer nest IS on self.P_mat to PC of main KSP. This allows
            # fieldsplit preconditioning to be applied, if desired.
            nest_IS = self.P_mat.getNestISs()
            fieldsplit_IS = tuple(
                [
                    (f"{u.name + '_' if u.name != 'f' else ''}{i}", IS)
                    for i, (u, IS) in enumerate(zip(self.u, nest_IS[0], strict=True))
                ]
            )
            self.solver.getKSP().getPC().setFieldSplitIS(*fieldsplit_IS)

    @property
    def bcs(self) -> tuple[DirichletBC, ...]:
        """Dirichlet boundary conditions applied to the problem.

        Assigning rebuilds the cached markers, diagonal data, block index
        sets and BC grouping, and re-registers both SNES callbacks.
        This is collective for MATIS matrices. Constrained dofs are fixed
        once a condition is built; values are re-read on every callback.
        """
        return self._bcs

    @bcs.setter
    def bcs(self, bcs: Sequence[DirichletBC] | None) -> None:
        conditions = tuple(bcs) if bcs is not None else ()
        bc_data = _matrix_bc_data(self.J, conditions)
        diagonal = _matrix_diag_data(bc_data, 1.0, _diag_on_ghost_rows(self.A, self.J))
        preconditioner_diagonal = (
            _matrix_diag_data(bc_data, 1.0, _diag_on_ghost_rows(self.P_mat, self.preconditioner))
            if self.preconditioner is not None and self.P_mat is not None
            else diagonal
        )
        index_sets = (
            _block_index_sets(bc_data)
            if isinstance(self.J, Sequence) and self.A.getType() != PETSc.Mat.Type.NEST
            else None
        )
        jacobian_data = _JacobianAssemblyData(
            bc_data, diagonal, preconditioner_diagonal, index_sets
        )
        if isinstance(self.J, Sequence):
            residual = self.F
            if not isinstance(residual, Sequence):
                raise ValueError("Expected a sequence of forms for a block/nest residual.")
            trial_bcs = _bcs_by_block(bc_data.column_spaces, conditions)
            residual_bcs = _bcs_by_block(_extract_function_spaces(residual), conditions)
        else:
            trial_bcs = [conditions]
            residual_bcs = conditions
        residual_data = _ResidualAssemblyData(
            bc_data.column_markers,
            _bc_lifting_values(bc_data.column_spaces, trial_bcs, PETSc.ScalarType),
            trial_bcs,
            residual_bcs,
        )
        jacobian_ctx = {
            "u": self.u,
            "jacobian": self.J,
            "preconditioner": self.preconditioner,
            "bcs": conditions,
            "_assembly_data": jacobian_data,
        }
        self.solver.setJacobian(assemble_jacobian, self.A, self.P_mat, kargs=jacobian_ctx)  # type: ignore[arg-type]
        function_ctx = {
            "u": self.u,
            "residual": self.F,
            "jacobian": self.J,
            "bcs": conditions,
            "_assembly_data": residual_data,
            "_blocks": self.b.getAttr("_blocks"),
        }
        self.solver.setFunction(assemble_residual, self.b, kargs=function_ctx)  # type: ignore[arg-type]
        old_data = getattr(self, "_jacobian_data", None)
        self._jacobian_data = jacobian_data
        if old_data is not None and old_data.index_sets is not None:
            for sets in old_data.index_sets:
                for index_set in sets:
                    index_set.destroy()
        self._bcs = conditions

    def set_update(self, update: typing.Callable[[int], None]) -> None:
        """Set a function called before each nonlinear iteration.

        Args:
            update: Function called with the index of the iteration that
                is about to be taken.
        """
        self.solver.setUpdate(lambda _snes, step: update(step))

    def solve(self) -> _U:
        """Solve the problem.

        This method updates the solution ``u`` function(s) stored in the
        problem instance.

        Note:
            The user is responsible for asserting convergence of the SNES
            solver e.g. ``assert problem.solver.getConvergedReason() > 0``.
            Alternatively, pass ``"snes_error_if_not_converged": True`` and
            ``"ksp_error_if_not_converged" : True`` in ``petsc_options`` to
            raise a ``PETScError`` on failure.

        Returns:
            The solution function(s).
        """
        # Copy current iterate into the work array.
        assign(self.u, self.x)

        # Solve problem
        self.solver.solve(None, self.x)
        dolfinx.la.petsc._ghost_update(self.x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore[arg-type]

        # Copy solution back to function
        assign(self.x, self.u)  # type: ignore

        return self.u

    def __del__(self) -> None:
        """Destroy PETSc objects created internally."""
        # __init__ may have raised before all attributes were set
        for name in ("_snes", "_A", "_b", "_x", "_P_mat"):
            if (obj := getattr(self, name, None)) is not None:
                obj.destroy()
        data = getattr(self, "_jacobian_data", None)
        if data is not None and data.index_sets is not None:
            for sets in data.index_sets:
                for index_set in sets:
                    index_set.destroy()

    @property
    def F(self) -> Form | Sequence[Form]:
        """The compiled residual."""
        return typing.cast(Form | Sequence[Form], self._F)

    @property
    def J(self) -> Form | Sequence[Sequence[Form]]:
        """The compiled Jacobian."""
        return typing.cast(Form | Sequence[Sequence[Form]], self._J)

    @property
    def preconditioner(self) -> Form | Sequence[Sequence[Form | None]] | None:
        """The compiled preconditioner."""
        return self._preconditioner

    @property
    def A(self) -> PETSc.Mat:
        """Jacobian matrix."""
        return self._A

    @property
    def P_mat(self) -> PETSc.Mat | None:
        """Preconditioner matrix."""
        return self._P_mat

    @property
    def b(self) -> PETSc.Vec:
        """Residual vector."""
        return self._b

    @property
    def x(self) -> PETSc.Vec:
        """Solution vector.

        Note:
            The vector does not share memory with the
            solution function(s) ``u``.
        """
        return self._x

    @property
    def solver(self) -> PETSc.SNES:
        """The SNES solver."""
        return self._snes

    @property
    def u(self) -> _U:
        """Solution function(s).

        Note:
            The function(s) do not share memory with the solution
            vector ``x``.
        """
        return self._u  # type: ignore[return-value]


# -- Additional free helper functions (interpolations, assignments etc.) --


def discrete_curl(space0: _FunctionSpace, space1: _FunctionSpace) -> PETSc.Mat:
    """Assemble a discrete curl operator.

    Args:
        space0: H1 space to interpolate the gradient from.
        space1: H(curl) space to interpolate into.

    Returns:
        Discrete curl operator.
    """
    return _discrete_curl(space0._cpp_object, space1._cpp_object)  # type: ignore[arg-type]


def discrete_gradient(space0: _FunctionSpace, space1: _FunctionSpace) -> PETSc.Mat:
    """Assemble a discrete gradient operator.

    The discrete gradient operator interpolates the gradient of a H1
    finite element function into a H(curl) space. It is assumed that the
    H1 space uses an identity map and the H(curl) space uses a covariant
    Piola map.

    Args:
        space0: H1 space to interpolate the gradient from.
        space1: H(curl) space to interpolate into.

    Returns:
        Discrete gradient operator.
    """
    return _discrete_gradient(space0._cpp_object, space1._cpp_object)  # type: ignore[arg-type]


def interpolation_matrix(V0: _FunctionSpace, V1: _FunctionSpace) -> PETSc.Mat:
    """Create an interpolation operator between finite element spaces.

    Consider is the vector of degrees-of-freedom  :math:`u_{i}`
    associated with a function in :math:`V_{i}`. This function returns
    the matrix :math:`\\Pi` sucht that

    .. math::

        u_{1} = \\Pi u_{0}.

    Args:
        V0: Space to interpolate from.
        V1: Space to interpolate into.

    Returns:
        The interpolation matrix :math:`\\Pi`.

    Note:
        The returned matrix is not finalised, i.e. ghost values are not
        accumulated.
    """  # noqa: D301
    return _interpolation_matrix(V0._cpp_object, V1._cpp_object)  # type: ignore[arg-type]


@functools.singledispatch
def _assign(u: object, x: object) -> None:
    """Assign :class:`Function` degrees-of-freedom to a vector.

    Assigns degree-of-freedom values in ``u``, which is possibly a
    sequence of ``Function``s, to ``x``. When ``u`` is a sequence of
    ``Function``s, degrees-of-freedom for the ``Function``s in ``u`` are
    'stacked' and assigned to ``x``. See :func:`assign` for
    documentation on how stacked assignment is handled.

    Args:
        u: ``Function`` (s) to assign degree-of-freedom value from.
        x: Vector to assign degree-of-freedom values in ``u`` to.
    """
    if not isinstance(x, PETSc.Vec):
        raise TypeError("Second argument must be a PETSc vector.")
    functions = typing.cast(_Function | Sequence[_Function], u)
    if x.getType() == PETSc.Vec.Type().NEST:
        if not isinstance(functions, Sequence):
            raise ValueError("A sequence of functions is required for a nested PETSc vector.")
        dolfinx.la.petsc.assign([v.x.array for v in functions], x)
    else:
        if isinstance(functions, Sequence):
            data0, data1 = [], []
            for v in functions:
                bs = v.function_space.dofmap.bs
                n = v.function_space.dofmap.index_map.size_local
                data0.append(v.x.array[: bs * n])
                data1.append(v.x.array[bs * n :])
            dolfinx.la.petsc.assign(data0 + data1, x)
        else:
            dolfinx.la.petsc.assign(functions.x.array, x)


@_assign.register
def _(x: PETSc.Vec, u: _Function | Sequence[_Function]) -> None:  # type: ignore[misc]
    """Assign vector entries to :class:`Function` degrees-of-freedom.

    Assigns values in ``x`` to the degrees-of-freedom of ``u``, which is
    possibly a Sequence of ``Function``s. When ``u`` is a Sequence of
    ``Function``s, values in ``x`` are assigned block-wise to the
    ``Function``s. See :func:`assign` for documentation on how blocked
    assignment is handled.

    Args:
        x: Vector with values to assign values from.
        u: ``Function`` (s) to assign degree-of-freedom values to.
    """
    if x.getType() == PETSc.Vec.Type().NEST:
        dolfinx.la.petsc.assign(x, [v.x.array for v in u])  # type: ignore
    else:
        if isinstance(u, Sequence):
            data0, data1 = [], []
            for v in u:
                bs = v.function_space.dofmap.bs
                n = v.function_space.dofmap.index_map.size_local
                data0.append(v.x.array[: bs * n])
                data1.append(v.x.array[bs * n :])
            dolfinx.la.petsc.assign(x, data0 + data1)  # type: ignore
        else:
            dolfinx.la.petsc.assign(x, u.x.array)  # type: ignore[bad-argument-type]


@overload
def assign(u: _Function | Sequence[_Function], x: PETSc.Vec) -> None: ...


@overload
def assign(u: PETSc.Vec, x: _Function | Sequence[_Function]) -> None: ...


def assign(
    u: _Function | Sequence[_Function] | PETSc.Vec,
    x: PETSc.Vec | _Function | Sequence[_Function],
) -> None:
    """Assign between function degrees-of-freedom and a PETSc vector."""
    _assign(u, x)


def get_petsc_lib() -> pathlib.Path:
    """Find the full path of the PETSc shared library.

    Returns:
        Full path to the PETSc shared library.

    Raises:
        RuntimeError: If PETSc library cannot be found.
    """
    import petsc4py as _petsc4py

    petsc_dir = _petsc4py.get_config()["PETSC_DIR"]
    petsc_arch = _petsc4py.lib.getPathArchPETSc()[1]  # type: ignore
    petsc_version = PETSc.Sys.getVersion()
    major_minor_version = ".".join(str(v) for v in petsc_version[:2])
    major_minor_patch_version = ".".join(str(v) for v in petsc_version[:3])
    candidate_paths = [
        os.path.join(petsc_dir, petsc_arch, "lib", f"libpetsc.so.{major_minor_patch_version}"),
        os.path.join(petsc_dir, petsc_arch, "lib", f"libpetsc.{major_minor_patch_version}.dylib"),
        os.path.join(petsc_dir, petsc_arch, "lib", f"libpetsc.so.{major_minor_version}"),
        os.path.join(petsc_dir, petsc_arch, "lib", f"libpetsc.{major_minor_version}.dylib"),
        os.path.join(petsc_dir, petsc_arch, "lib", "libpetsc.so"),
        os.path.join(petsc_dir, petsc_arch, "lib", "libpetsc.dylib"),
    ]
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return pathlib.Path(candidate_path)

    raise RuntimeError(f"Could not find a PETSc shared library. Candidate paths: {candidate_paths}")


class numba_utils:
    """Utility attributes for working with Numba and PETSc.

    These attributes are convenience functions for calling PETSc C
    functions from within Numba functions.

    Note:
        `Numba <https://numba.pydata.org/>`_ must be available
        to use these utilities.

    Examples:
        A typical use of these utility functions is::

            import numpy as np
            import numpy.typing as npt
            def set_vals(A: int,
                         m: int, rows: npt.NDArray[PETSc.IntType],
                         n: int, cols: npt.NDArray[PETSc.IntType],
                         data: npt.NDArray[PETSc.ScalarTYpe], mode: int):
                MatSetValuesLocal(A, m, rows.ctypes, n, cols.ctypes,
                                  data.ctypes, mode)
    """

    try:
        import petsc4py.PETSc as _PETSc

        import llvmlite as _llvmlite
        import numba as _numba

        _llvmlite.binding.load_library_permanently(str(get_petsc_lib()))

        _int = _numba.from_dtype(_PETSc.IntType)
        _scalar = _numba.from_dtype(_PETSc.ScalarType)
        _real = _numba.from_dtype(_PETSc.RealType)
        _int_ptr = _numba.core.types.CPointer(_int)
        _scalar_ptr = _numba.core.types.CPointer(_scalar)
        _MatSetValues_sig = _numba.core.typing.signature(
            _numba.core.types.intc,
            _numba.core.types.uintp,
            _int,
            _int_ptr,
            _int,
            _int_ptr,
            _scalar_ptr,
            _numba.core.types.intc,
        )
        MatSetValuesLocal = _numba.core.types.ExternalFunction(
            "MatSetValuesLocal", _MatSetValues_sig
        )
        """See PETSc `MatSetValuesLocal
        <https://petsc.org/release/manualpages/Mat/MatSetValuesLocal>`_
        documentation."""

        MatSetValuesBlockedLocal = _numba.core.types.ExternalFunction(
            "MatSetValuesBlockedLocal", _MatSetValues_sig
        )
        """See PETSc `MatSetValuesBlockedLocal
        <https://petsc.org/release/manualpages/Mat/MatSetValuesBlockedLocal>`_
        documentation."""
    except ImportError:
        # numba/llvmlite/petsc4py not installed; numba bindings unavailable
        pass


class ctypes_utils:
    """Utility attributes for working with ctypes and PETSc.

    These attributes are convenience functions for calling PETSc C
    functions, typically from within Numba functions.

    Examples:
        A typical use of these utility functions is::

            import numpy as np
            import numpy.typing as npt
            def set_vals(A: int,
                         m: int, rows: npt.NDArray[PETSc.IntType],
                         n: int, cols: npt.NDArray[PETSc.IntType],
                         data: npt.NDArray[PETSc.ScalarTYpe], mode: int):
                MatSetValuesLocal(A, m, rows.ctypes, n, cols.ctypes,
                                  data.ctypes, mode)
    """

    try:
        import petsc4py.PETSc as _PETSc

        _lib_ctypes = _ctypes.cdll.LoadLibrary(str(get_petsc_lib()))

        # Note: ctypes does not have complex types, hence we use void* for
        # scalar data
        _int = np.ctypeslib.as_ctypes_type(_PETSc.IntType)

        MatSetValuesLocal = _lib_ctypes.MatSetValuesLocal
        """See PETSc `MatSetValuesLocal
        <https://petsc.org/release/manualpages/Mat/MatSetValuesLocal>`_
        documentation."""
        MatSetValuesLocal.argtypes = [
            _ctypes.c_void_p,
            _int,
            _ctypes.POINTER(_int),
            _int,
            _ctypes.POINTER(_int),
            _ctypes.c_void_p,
            _ctypes.c_int,
        ]

        MatSetValuesBlockedLocal = _lib_ctypes.MatSetValuesBlockedLocal
        """See PETSc `MatSetValuesBlockedLocal
        <https://petsc.org/release/manualpages/Mat/MatSetValuesBlockedLocal>`_
        documentation."""
        MatSetValuesBlockedLocal.argtypes = [
            _ctypes.c_void_p,
            _int,
            _ctypes.POINTER(_int),
            _int,
            _ctypes.POINTER(_int),
            _ctypes.c_void_p,
            _ctypes.c_int,
        ]
    except ImportError:
        # petsc4py not installed; ctypes bindings unavailable
        pass


class cffi_utils:
    """Utility attributes for working with CFFI (ABI mode) and Numba.

    Registers Numba's complex types with CFFI.

    If PETSc is available, CFFI convenience functions for calling PETSc C
    functions are also created. These are typically called from within
    Numba functions.

    Note:
        `CFFI <https://cffi.readthedocs.io/>`_ and  `Numba
        <https://numba.pydata.org/>`_ must be available to use these
        utilities.

    Examples:
        A typical use of these utility functions is::

            import numpy as np
            import numpy.typing as npt
            def set_vals(A: int,
                         m: int, rows: npt.NDArray[PETSc.IntType],
                         n: int, cols: npt.NDArray[PETSc.IntType],
                         data: npt.NDArray[PETSc.ScalarType], mode: int):
                MatSetValuesLocal(A, m, ffi.from_buffer(rows), n,
                                  ffi.from_buffer(cols),
                                  ffi.from_buffer(rows(data), mode)
    """

    import cffi as _cffi

    _ffi = _cffi.FFI()

    try:
        import numba as _numba
        import numba.core.typing.cffi_utils as _cffi_support

        # Register complex types
        _cffi_support.register_type(_ffi.typeof("float _Complex"), _numba.types.complex64)
        _cffi_support.register_type(_ffi.typeof("double _Complex"), _numba.types.complex128)

    except KeyError:
        # complex types already registered with numba/cffi
        pass
    except ImportError:
        log(
            LogLevel.DEBUG,
            "Could not import numba, so cffi/numba complex types were not registered.",
        )

    try:
        from petsc4py import PETSc as _PETSc

        _lib_cffi = _ffi.dlopen(str(get_petsc_lib()))

        _CTYPES = {
            np.int32: "int32_t",
            np.int64: "int64_t",
            np.float32: "float",
            np.float64: "double",
            np.complex64: "float _Complex",
            np.complex128: "double _Complex",
            np.longlong: "long long",
        }

        _c_int_t = _CTYPES[_PETSc.IntType]  # type: ignore
        _c_scalar_t = _CTYPES[_PETSc.ScalarType]  # type: ignore
        _ffi.cdef(
            f"""
                int MatSetValuesLocal(void* mat, {_c_int_t} nrow, const {_c_int_t}* irow,
                                    {_c_int_t} ncol, const {_c_int_t}* icol,
                                    const {_c_scalar_t}* y, int addv);
                int MatSetValuesBlockedLocal(void* mat, {_c_int_t} nrow, const {_c_int_t}* irow,
                                    {_c_int_t} ncol, const {_c_int_t}* icol,
                                    const {_c_scalar_t}* y, int addv);
                                    """
        )

        MatSetValuesLocal = _lib_cffi.MatSetValuesLocal
        """See PETSc `MatSetValuesLocal
        <https://petsc.org/release/manualpages/Mat/MatSetValuesLocal>`_
        documentation."""

        MatSetValuesBlockedLocal = _lib_cffi.MatSetValuesBlockedLocal
        """See PETSc `MatSetValuesBlockedLocal
        <https://petsc.org/release/manualpages/Mat/MatSetValuesBlockedLocal>`_
        documentation."""
    except KeyError:
        # PETSc scalar/index type has no corresponding C type in _CTYPES
        pass
    except ImportError:
        log(
            LogLevel.DEBUG,
            "Could not import petsc4py, so cffi/PETSc ABI mode interface was not created.",
        )
