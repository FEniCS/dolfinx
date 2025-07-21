# Copyright (C) 2018-2025 Garth N. Wells, Nathan Sime and Jørgen S. Dokken
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
import functools
import typing
from collections.abc import Sequence

from petsc4py import PETSc

# ruff: noqa: E402
import dolfinx

assert dolfinx.has_petsc4py

import warnings
from functools import partial

import numpy as np
from numpy import typing as npt

import dolfinx.cpp as _cpp
import dolfinx.la.petsc
import ufl
from dolfinx.cpp.fem.petsc import discrete_curl as _discrete_curl
from dolfinx.cpp.fem.petsc import discrete_gradient as _discrete_gradient
from dolfinx.cpp.fem.petsc import interpolation_matrix as _interpolation_matrix
from dolfinx.fem import IntegralType, pack_coefficients, pack_constants
from dolfinx.fem.assemble import _assemble_vector_array
from dolfinx.fem.assemble import apply_lifting as _apply_lifting
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
    "NewtonSolverNonlinearProblem",
    "NonlinearProblem",
    "apply_lifting",
    "assemble_jacobian",
    "assemble_matrix",
    "assemble_residual",
    "assemble_vector",
    "assign",
    "create_matrix",
    "create_vector",
    "discrete_curl",
    "discrete_gradient",
    "interpolation_matrix",
    "set_bc",
]


# -- Vector instantiation -------------------------------------------------


def create_vector(
    container: typing.Union[Form, _FunctionSpace, Sequence[Form], Sequence[_FunctionSpace]],
    /,
    kind: typing.Optional[str] = None,
) -> PETSc.Vec:  # type: ignore[name-defined]
    """Create a PETSc vector that is compatible with a linear form(s)
    or functionspace(s).

    Three cases are supported:

    1. For a single linear form ``L`` or space ``V``, if ``kind`` is
       ``None`` or is ``PETSc.Vec.Type.MPI``, a ghosted PETSc vector which
       is compatible with ``L/V`` is created.

    2. If ``L/V`` is a sequence of linear forms/functionspaces and ``kind``
       is ``None`` or is ``PETSc.Vec.Type.MPI``, a ghosted PETSc vector
       which is compatible with ``L`` is created. The created vector ``b``
       is initialized such that on each MPI process ``b = [b_0, b_1, ...,
       b_n, b_0g, b_1g, ..., b_ng]``, where ``b_i`` are the entries
       associated with the 'owned' degrees-of-freedom for ``L[i]`` and
       ``b_ig`` are the 'unowned' (ghost) entries for ``L[i]``.

       For this case, the returned vector has an attribute ``_blocks``
       that holds the local offsets into ``b`` for the (i) owned and
       (ii) ghost entries for each ``L_i/V_i``. It can be accessed by
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
        : Linear form/function space or a sequence of such.
        kind: PETSc vector type (``VecType``) to create.

    Returns:
        A PETSc vector with a layout that is compatible with ``L``. The
        vector is not initialised to zero.
    """
    if isinstance(container, (Form, _FunctionSpace)):
        container = [container]

    if len(container) == 0:
        raise RuntimeError("Empty sequence of functionspaces/forms provided.")

    V = _extract_function_spaces(container) if isinstance(container[0], Form) else container

    if any(_V is None for _V in V):
        raise RuntimeError("Can not create vector for extracted None block.")

    maps = [(_V.dofmap.index_map, _V.dofmap.index_map_bs) for _V in V if _V is not None]
    return dolfinx.la.petsc.create_vector(maps, kind=kind)


# -- Matrix instantiation -------------------------------------------------


def create_matrix(
    a: typing.Union[Form, Sequence[Sequence[Form]]],
    kind: typing.Optional[typing.Union[str, Sequence[Sequence[str]]]] = None,
) -> PETSc.Mat:  # type: ignore[name-defined]
    """Create a PETSc matrix that is compatible with the (sequence) of
    bilinear form(s).

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
       array of for forms ``a``. If ``kind`` is ``None``, then the
       matrix is the default type.

       In this case, the matrix is arranged::

             A = [a_00 ... a_0n]
                 [a_10 ... a_1n]
                 [     ...     ]
                 [a_m0 ..  a_mn]

    Args:
        a: A bilinear form or a nested sequence of bilinear forms.
        kind: The PETSc matrix type (``MatType``).

    Returns:
        A PETSc matrix.
    """
    if isinstance(a, Sequence):
        _a = [[None if form is None else form._cpp_object for form in arow] for arow in a]
        if kind == PETSc.Mat.Type.NEST:  # type: ignore[attr-defined]
            # Create nest matrix with default types
            return _cpp.fem.petsc.create_matrix_nest(_a, None)
        else:
            if kind is None or isinstance(kind, str):  # Single 'kind' type
                return _cpp.fem.petsc.create_matrix_block(_a, kind)
            else:  # Array of 'kind' types
                return _cpp.fem.petsc.create_matrix_nest(_a, kind)
    else:  # Single form
        return _cpp.fem.petsc.create_matrix(a._cpp_object, kind)


# -- Vector assembly ------------------------------------------------------


@functools.singledispatch
def assemble_vector(
    L: typing.Union[Form, Sequence[Form]],
    constants: typing.Optional[typing.Union[npt.NDArray, Sequence[npt.NDArray]]] = None,
    coeffs: typing.Optional[
        typing.Union[
            dict[tuple[IntegralType, int], npt.NDArray],
            Sequence[dict[tuple[IntegralType, int], npt.NDArray]],
        ]
    ] = None,
    kind: typing.Optional[str] = None,
) -> PETSc.Vec:  # type: ignore[name-defined]
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
       ``L[i]`` is assembled into into the ith nested matrix in ``b``.

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
    b = create_vector(L, kind=kind)
    dolfinx.la.petsc._zero_vector(b)
    return assemble_vector(b, L, constants, coeffs)  # type: ignore[arg-type]


@assemble_vector.register
def _(
    b: PETSc.Vec,  # type: ignore[name-defined]
    L: typing.Union[Form, Sequence[Form]],
    constants: typing.Optional[typing.Union[npt.NDArray, Sequence[npt.NDArray]]] = None,
    coeffs: typing.Optional[
        typing.Union[
            dict[tuple[IntegralType, int], npt.NDArray],
            Sequence[dict[tuple[IntegralType, int], npt.NDArray]],
        ]
    ] = None,
) -> PETSc.Vec:  # type: ignore[name-defined]
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
    if b.getType() == PETSc.Vec.Type.NEST:  # type: ignore[attr-defined]
        if not isinstance(L, Sequence):
            raise ValueError("Must provide a sequence of forms when assembling a nest vector")
        constants = [None] * len(L) if constants is None else constants  # type: ignore[list-item]
        coeffs = [None] * len(L) if coeffs is None else coeffs  # type: ignore[list-item]
        for b_sub, L_sub, const, coeff in zip(b.getNestSubVecs(), L, constants, coeffs):
            with b_sub.localForm() as b_local:
                _assemble_vector_array(b_local.array_w, L_sub, const, coeff)  # type: ignore[arg-type]
    elif isinstance(L, Sequence):
        constants = pack_constants(L) if constants is None else constants
        coeffs = pack_coefficients(L) if coeffs is None else coeffs
        offset0, offset1 = b.getAttr("_blocks")
        with b.localForm() as b_l:
            for L_, const, coeff, off0, off1, offg0, offg1 in zip(
                L, constants, coeffs, offset0, offset0[1:], offset1, offset1[1:]
            ):
                bx_ = np.zeros((off1 - off0) + (offg1 - offg0), dtype=PETSc.ScalarType)  # type: ignore[attr-defined]
                _assemble_vector_array(bx_, L_, const, coeff)  # type: ignore[arg-type]
                size = off1 - off0
                b_l.array_w[off0:off1] += bx_[:size]
                b_l.array_w[offg0:offg1] += bx_[size:]
    else:
        with b.localForm() as b_local:
            _assemble_vector_array(b_local.array_w, L, constants, coeffs)  # type: ignore[arg-type]

    return b


# -- Matrix assembly ------------------------------------------------------
@functools.singledispatch
def assemble_matrix(
    a: typing.Union[Form, Sequence[Sequence[Form]]],
    bcs: typing.Optional[Sequence[DirichletBC]] = None,
    diag: float = 1,
    constants: typing.Optional[
        typing.Union[Sequence[npt.NDArray], Sequence[Sequence[npt.NDArray]]]
    ] = None,
    coeffs: typing.Optional[
        typing.Union[
            dict[tuple[IntegralType, int], npt.NDArray],
            Sequence[dict[tuple[IntegralType, int], npt.NDArray]],
        ]
    ] = None,
    kind=None,
):
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
          ``kind`` of the default type (if ``kind`` is ``None``).
       #. If ``kind`` is ``PETSc.Mat.Type.NEST`` or a rectangular array
          of PETSc matrix types, the returned matrix has type
          ``PETSc.Mat.Type.NEST``.

    Rows/columns that are constrained by a Dirichlet boundary condition
    are zeroed, with the diagonal to set to ``diag``.

    Constant and coefficient data that appear in the forms(s) can be
    packed outside of this function to avoid re-packing by this
    function. The functions :func:`dolfinx.fem.pack_constants` and
    :func:`dolfinx.fem.pack_coefficients` can be used to 'pre-pack' the
    data.

    Note:
        The returned matrix is not 'assembled', i.e. ghost contributions
        are not accumulated.

    Args:
        a: Bilinear form(s) to assembled into a matrix.
        bc: Dirichlet boundary conditions applied to the system.
        diag: Value to set on the matrix diagonal for Dirichlet
            boundary condition constrained degrees-of-freedom belonging
            to the same trial and test space.
        constants: Constants appearing the in the form.
        coeffs: Coefficients appearing the in the form.

    Returns:
        Matrix representing the bilinear form.
    """
    A = create_matrix(a, kind)
    assemble_matrix(A, a, bcs, diag, constants, coeffs)  # type: ignore[arg-type]
    return A


@assemble_matrix.register
def _(
    A: PETSc.Mat,  # type: ignore[name-defined]
    a: typing.Union[Form, Sequence[Sequence[Form]]],
    bcs: typing.Optional[Sequence[DirichletBC]] = None,
    diag: float = 1,
    constants: typing.Optional[typing.Union[npt.NDArray, Sequence[Sequence[npt.NDArray]]]] = None,
    coeffs: typing.Optional[
        typing.Union[
            dict[tuple[IntegralType, int], npt.NDArray],
            Sequence[Sequence[dict[tuple[IntegralType, int], npt.NDArray]]],
        ]
    ] = None,
) -> PETSc.Mat:  # type: ignore[name-defined]
    """Assemble bilinear form into a matrix.

    The matrix vector ``A`` must have been initialized with a
    size/layout that is consistent with the bilinear form(s). The PETSc
    matrix ``A`` is normally created by :func:`create_matrix`.

    The returned matrix is not finalised, i.e. ghost values are not
    accumulated.
    """
    if A.getType() == PETSc.Mat.Type.NEST:  # type: ignore[attr-defined]
        if not isinstance(a, Sequence):
            raise ValueError("Must provide a sequence of forms when assembling a nest matrix")
        constants = [pack_constants(forms) for forms in a] if constants is None else constants  # type: ignore[misc]
        coeffs = [pack_coefficients(forms) for forms in a] if coeffs is None else coeffs  # type: ignore[misc]
        for i, (a_row, const_row, coeff_row) in enumerate(zip(a, constants, coeffs)):
            for j, (a_block, const, coeff) in enumerate(zip(a_row, const_row, coeff_row)):
                if a_block is not None:
                    Asub = A.getNestSubMatrix(i, j)
                    assemble_matrix(Asub, a_block, bcs, diag, const, coeff)  # type: ignore[arg-type]
                elif i == j:
                    for bc in bcs:
                        row_forms = [row_form for row_form in a_row if row_form is not None]
                        assert len(row_forms) > 0
                        if row_forms[0].function_spaces[0].contains(bc.function_space):
                            raise RuntimeError(
                                f"Diagonal sub-block ({i}, {j}) cannot be 'None'"
                                " and have DirichletBC applied."
                                " Consider assembling a zero block."
                            )
    elif isinstance(a, Sequence):  # Blocked
        consts = [pack_constants(forms) for forms in a] if constants is None else constants  # type: ignore[misc]
        coeffs = [pack_coefficients(forms) for forms in a] if coeffs is None else coeffs  # type: ignore[misc]
        V = (_extract_function_spaces(a, 0), _extract_function_spaces(a, 1))
        for index in range(2):
            # the check below is to ensure that a .dofmaps attribute is
            # available when creating is0 and is1 below
            if all(Vsub is None for Vsub in V[index]):
                raise ValueError(
                    "Cannot have a entire {'row' if index == 0 else 'column'} of a full of None"
                )
        is0 = _cpp.la.petsc.create_index_sets(
            [(Vsub.dofmaps(0).index_map, Vsub.dofmaps(0).index_map_bs) for Vsub in V[0]]  # type: ignore[union-attr]
        )
        is1 = _cpp.la.petsc.create_index_sets(
            [(Vsub.dofmaps(0).index_map, Vsub.dofmaps(0).index_map_bs) for Vsub in V[1]]  # type: ignore[union-attr]
        )

        _bcs = [bc._cpp_object for bc in bcs] if bcs is not None else []
        for i, a_row in enumerate(a):
            for j, a_sub in enumerate(a_row):
                if a_sub is not None:
                    Asub = A.getLocalSubMatrix(is0[i], is1[j])
                    _cpp.fem.petsc.assemble_matrix(
                        Asub,
                        a_sub._cpp_object,
                        consts[i][j],
                        coeffs[i][j],  # type: ignore[index]
                        _bcs,
                        True,
                    )
                    A.restoreLocalSubMatrix(is0[i], is1[j], Asub)
                elif i == j:
                    for bc in _bcs:
                        row_forms = [row_form for row_form in a_row if row_form is not None]
                        assert len(row_forms) > 0
                        if row_forms[0].function_spaces[0].contains(bc.function_space):
                            raise RuntimeError(
                                f"Diagonal sub-block ({i}, {j}) cannot be 'None' "
                                " and have DirichletBC applied."
                                " Consider assembling a zero block."
                            )

        # Flush to enable switch from add to set in the matrix
        A.assemble(PETSc.Mat.AssemblyType.FLUSH)  # type: ignore[attr-defined]

        # Set diagonal
        for i, a_row in enumerate(a):
            for j, a_sub in enumerate(a_row):
                if a_sub is not None:
                    Asub = A.getLocalSubMatrix(is0[i], is1[j])
                    if a_sub.function_spaces[0] is a_sub.function_spaces[1]:
                        _cpp.fem.petsc.insert_diagonal(Asub, a_sub.function_spaces[0], _bcs, diag)
                    A.restoreLocalSubMatrix(is0[i], is1[j], Asub)
    else:  # Non-blocked
        constants = pack_constants(a) if constants is None else constants  # type: ignore[assignment]
        coeffs = pack_coefficients(a) if coeffs is None else coeffs  # type: ignore[assignment]
        _bcs = [bc._cpp_object for bc in bcs] if bcs is not None else []
        _cpp.fem.petsc.assemble_matrix(A, a._cpp_object, constants, coeffs, _bcs)
        if a.function_spaces[0] is a.function_spaces[1]:
            A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)  # type: ignore[attr-defined]
            A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)  # type: ignore[attr-defined]
            _cpp.fem.petsc.insert_diagonal(A, a.function_spaces[0], _bcs, diag)

    return A


# -- Modifiers for Dirichlet conditions -----------------------------------


def apply_lifting(
    b: PETSc.Vec,  # type: ignore[name-defined]
    a: typing.Union[Sequence[Form], Sequence[Sequence[Form]]],
    bcs: typing.Optional[typing.Union[Sequence[DirichletBC], Sequence[Sequence[DirichletBC]]]],
    x0: typing.Optional[Sequence[PETSc.Vec]] = None,  # type: ignore[name-defined]
    alpha: float = 1,
    constants: typing.Optional[
        typing.Union[Sequence[npt.NDArray], Sequence[Sequence[npt.NDArray]]]
    ] = None,
    coeffs: typing.Optional[
        typing.Union[
            dict[tuple[IntegralType, int], npt.NDArray],
            Sequence[Sequence[dict[tuple[IntegralType, int], npt.NDArray]]],
        ]
    ] = None,
) -> None:
    r"""Modify the right-hand side PETSc vector ``b`` to account for
    constraints (Dirichlet boundary conitions).

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
    """
    if b.getType() == PETSc.Vec.Type.NEST:  # type: ignore[attr-defined]
        x0 = [] if x0 is None else x0.getNestSubVecs()  # type: ignore[attr-defined]
        constants = [pack_constants(forms) for forms in a] if constants is None else constants  # type: ignore[assignment]
        coeffs = [pack_coefficients(forms) for forms in a] if coeffs is None else coeffs  # type: ignore[misc]
        for b_sub, a_sub, const, coeff in zip(b.getNestSubVecs(), a, constants, coeffs):  # type: ignore[arg-type]
            const_ = list(
                map(lambda x: np.array([], dtype=PETSc.ScalarType) if x is None else x, const)  # type: ignore[attr-defined, call-overload]
            )
            apply_lifting(b_sub, a_sub, bcs, x0, alpha, const_, coeff)  # type: ignore[arg-type]
    else:
        with contextlib.ExitStack() as stack:
            if b.getAttr("_blocks") is not None:
                if x0 is not None:
                    offset0, offset1 = x0.getAttr("_blocks")  # type: ignore[attr-defined]
                    xl = stack.enter_context(x0.localForm())  # type: ignore[attr-defined]
                    xlocal = [
                        np.concatenate((xl[off0:off1], xl[offg0:offg1]))
                        for (off0, off1, offg0, offg1) in zip(
                            offset0, offset0[1:], offset1, offset1[1:]
                        )
                    ]
                else:
                    xlocal = None

                offset0, offset1 = b.getAttr("_blocks")
                with b.localForm() as b_l:
                    for i, (a_, off0, off1, offg0, offg1) in enumerate(
                        zip(a, offset0, offset0[1:], offset1, offset1[1:])
                    ):
                        const = pack_constants(a_) if constants is None else constants[i]  # type: ignore[arg-type]
                        coeff = pack_coefficients(a_) if coeffs is None else coeffs[i]  # type: ignore[arg-type, assignment, index]
                        const_ = [
                            np.empty(0, dtype=PETSc.ScalarType) if val is None else val  # type: ignore[attr-defined]
                            for val in const
                        ]
                        bx_ = np.concatenate((b_l[off0:off1], b_l[offg0:offg1]))
                        _apply_lifting(bx_, a_, bcs, xlocal, float(alpha), const_, coeff)  # type: ignore[arg-type]
                        size = off1 - off0
                        b_l.array_w[off0:off1] = bx_[:size]
                        b_l.array_w[offg0:offg1] = bx_[size:]
            else:
                x0 = [] if x0 is None else x0
                x0 = [stack.enter_context(x.localForm()) for x in x0]
                x0_r = [x.array_r for x in x0]
                b_local = stack.enter_context(b.localForm())
                _apply_lifting(b_local.array_w, a, bcs, x0_r, alpha, constants, coeffs)  # type: ignore[arg-type]

    return b


def set_bc(
    b: PETSc.Vec,  # type: ignore[name-defined]
    bcs: typing.Union[Sequence[DirichletBC], Sequence[Sequence[DirichletBC]]],
    x0: typing.Optional[PETSc.Vec] = None,  # type: ignore[name-defined]
    alpha: float = 1,
) -> None:
    r"""Set constraint (Dirchlet boundary condition) values in an vector.

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
            ``bcs`` should be a sequence of ``DirichletBC``\s. For
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
        x0 = x0.array_r if x0 is not None else None
        for bc in bcs:
            bc.set(b.array_w, x0, alpha)  # type: ignore[union-attr]
    elif b.getType() == PETSc.Vec.Type.NEST:  # type: ignore[attr-defined]
        _b = b.getNestSubVecs()
        x0 = len(_b) * [None] if x0 is None else x0.getNestSubVecs()
        for b_sub, bc, x_sub in zip(_b, bcs, x0):  # type: ignore[assignment, arg-type]
            set_bc(b_sub, bc, x_sub, alpha)  # type: ignore[arg-type]
    else:  # block vector
        offset0, _ = b.getAttr("_blocks")
        b_array = b.getArray(readonly=False)
        x_array = x0.getArray(readonly=True) if x0 is not None else None
        for bcs, off0, off1 in zip(bcs, offset0, offset0[1:]):  # type: ignore[assignment]
            x0_sub = x_array[off0:off1] if x0 is not None else None  # type: ignore[index]
            for bc in bcs:
                bc.set(b_array[off0:off1], x0_sub, alpha)  # type: ignore[arg-type, union-attr]


# -- High-level interface for KSP ---------------------------------------


class LinearProblem:
    """High-level class for solving a linear variational problem using
    a PETSc KSP.

    Solves problems of the form
    :math:`a_{ij}(u, v) = f_i(v), i,j=0,\\ldots,N\\
    \\forall v \\in V` where
    :math:`u=(u_0,\\ldots,u_N), v=(v_0,\\ldots,v_N)`
    using PETSc KSP as the linear solver.

    Note:
        This high-level class automatically handles PETSc memory
        management. The user does not need to manually call
        ``.destroy()`` on returned PETSc objects.
    """

    def __init__(
        self,
        a: typing.Union[ufl.Form, Sequence[Sequence[ufl.Form]]],
        L: typing.Union[ufl.Form, Sequence[ufl.Form]],
        *,
        petsc_options_prefix: str,
        bcs: typing.Optional[Sequence[DirichletBC]] = None,
        u: typing.Optional[typing.Union[_Function, Sequence[_Function]]] = None,
        P: typing.Optional[typing.Union[ufl.Form, Sequence[Sequence[ufl.Form]]]] = None,
        kind: typing.Optional[typing.Union[str, Sequence[Sequence[str]]]] = None,
        petsc_options: typing.Optional[dict] = None,
        form_compiler_options: typing.Optional[dict] = None,
        jit_options: typing.Optional[dict] = None,
        entity_maps: typing.Optional[Sequence[_EntityMap]] = None,
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
                forms, used as a preconditioner.
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
                as the integration domain, ``entity_maps`` must be
                supplied.
                For each key (a mesh, different to the integration domain
                mesh) a map should be provided relating the entities in the
                integration domain mesh to the entities in the key mesh
                e.g. for a key-value pair (msh, emap) in ``entity_maps``,
                ``emap[i]`` is the entity in ``msh`` corresponding to
                entity ``i`` in the integration domain mesh.
        """
        self._a = _create_form(
            a,
            dtype=PETSc.ScalarType,  # type: ignore[attr-defined]
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )
        self._L = _create_form(
            L,
            dtype=PETSc.ScalarType,  # type: ignore[attr-defined]
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )
        self._A = create_matrix(self._a, kind=kind)
        self._preconditioner = _create_form(
            P,  # type: ignore[arg-type]
            dtype=PETSc.ScalarType,  # type: ignore[attr-defined]
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )
        self._P_mat = (
            create_matrix(self._preconditioner, kind=kind)
            if self._preconditioner is not None
            else None
        )

        # For nest matrices kind can be a nested list.
        kind = "nest" if self.A.getType() == PETSc.Mat.Type.NEST else kind  # type: ignore[attr-defined]
        assert kind is None or isinstance(kind, str)
        self._b = create_vector(self.L, kind=kind)
        self._x = create_vector(self.L, kind=kind)

        self._u: typing.Union[_Function, Sequence[_Function]]
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

        self._solver = PETSc.KSP().create(self.A.comm)  # type: ignore[attr-defined]
        self.solver.setOperators(self.A, self.P_mat)

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
            opts = PETSc.Options()  # type: ignore[attr-defined]
            opts.prefixPush(self.solver.getOptionsPrefix())

            for k, v in petsc_options.items():
                opts[k] = v

            self.solver.setFromOptions()

            # Tidy up global options
            for k in petsc_options.keys():
                del opts[k]

            opts.prefixPop()

        if self.P_mat is not None and kind == "nest":
            # Transfer nest IS on self.P_mat to PC of main KSP. This allows
            # fieldsplit preconditioning to be applied, if desired.
            nest_IS = self.P_mat.getNestISs()
            fieldsplit_IS = tuple(
                [
                    (f"{u.name + '_' if u.name != 'f' else ''}{i}", IS)
                    for i, (u, IS) in enumerate(zip(self.u, nest_IS[0]))
                ]
            )
            self.solver.getPC().setFieldSplitIS(*fieldsplit_IS)

    def __del__(self):
        self._solver.destroy()
        self._A.destroy()
        self._b.destroy()
        self._x.destroy()
        if self._P_mat is not None:
            self._P_mat.destroy()

    def solve(self) -> typing.Union[_Function, Sequence[_Function]]:
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
        assemble_matrix(self.A, self.a, bcs=self.bcs)  # type: ignore[arg-type, misc]
        self.A.assemble()

        # Assemble preconditioner
        if self.P_mat is not None:
            self.P_mat.zeroEntries()
            assemble_matrix(self.P_mat, self.preconditioner, bcs=self.bcs)  # type: ignore[arg-type, misc]
            self.P_mat.assemble()

        # Assemble rhs
        dolfinx.la.petsc._zero_vector(self.b)
        assemble_vector(self.b, self.L)  # type: ignore[arg-type]

        # Apply boundary conditions to the rhs
        if self.bcs is not None:
            if isinstance(self.u, Sequence):  # block or nest
                bcs1 = _bcs_by_block(_extract_function_spaces(self.a, 1), self.bcs)  # type: ignore[arg-type]
                apply_lifting(self.b, self.a, bcs=bcs1)  # type: ignore[arg-type]
                dolfinx.la.petsc._ghost_update(
                    self.b,
                    PETSc.InsertMode.ADD,  # type: ignore[attr-defined]
                    PETSc.ScatterMode.REVERSE,  # type: ignore[attr-defined]
                )
                bcs0 = _bcs_by_block(_extract_function_spaces(self.L), self.bcs)  # type: ignore[arg-type]
                dolfinx.fem.petsc.set_bc(self.b, bcs0)
            else:  # single form
                apply_lifting(self.b, [self.a], bcs=[self.bcs])  # type: ignore[arg-type]
                dolfinx.la.petsc._ghost_update(
                    self.b,
                    PETSc.InsertMode.ADD,  # type: ignore[attr-defined]
                    PETSc.ScatterMode.REVERSE,  # type: ignore[attr-defined]
                )
                for bc in self.bcs:
                    bc.set(self.b.array_w)
        else:
            dolfinx.la.petsc._ghost_update(self.b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]

        # Solve linear system and update ghost values in the solution
        self.solver.solve(self.b, self.x)
        dolfinx.la.petsc._ghost_update(self.x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore[attr-defined]
        dolfinx.fem.petsc.assign(self.x, self.u)
        return self.u

    @property
    def L(self) -> typing.Union[Form, Sequence[Form]]:
        """The compiled linear form representing the left-hand side."""
        return self._L

    @property
    def a(self) -> typing.Union[Form, Sequence[Form]]:
        """The compiled bilinear form representing the right-hand side."""
        return self._a

    @property
    def preconditioner(self) -> typing.Union[Form, Sequence[Form]]:
        """The compiled bilinear form representing the preconditioner."""
        return self._preconditioner

    @property
    def A(self) -> PETSc.Mat:  # type: ignore[name-defined]
        """Left-hand side matrix."""
        return self._A

    @property
    def P_mat(self) -> PETSc.Mat:  # type: ignore[name-defined]
        """Preconditioner matrix."""
        return self._P_mat

    @property
    def b(self) -> PETSc.Vec:  # type: ignore[name-defined]
        """Right-hand side vector."""
        return self._b

    @property
    def x(self) -> PETSc.Vec:  # type: ignore[name-defined]
        """Solution vector.

        Note:
            The vector does not share memory with the solution
            function(s) ``u``.
        """
        return self._x

    @property
    def solver(self) -> PETSc.KSP:  # type: ignore[name-defined]
        """The PETSc KSP solver."""
        return self._solver

    @property
    def u(self) -> typing.Union[_Function, Sequence[_Function]]:
        """Solution function(s).

        Note:
            The function(s) do not share memory with the solution
            vector ``x``.
        """
        return self._u


# -- High-level interface for SNES ---------------------------------------


def _assign_block_data(forms: Sequence[Form], vec: PETSc.Vec):  # type: ignore[name-defined]
    """Assign block data to a PETSc vector.

    Args:
        forms: List of forms to extract block data from.
        vec: PETSc vector to assign block data to.
    """

    maps = (
        (
            form.function_spaces[0].dofmaps(0).index_map,  # type: ignore[attr-defined]
            form.function_spaces[0].dofmaps(0).index_map_bs,  # type: ignore[attr-defined]
        )
        for form in forms
    )

    return dolfinx.la.petsc._assign_block_data(maps, vec)


def assemble_residual(
    u: typing.Union[_Function, Sequence[_Function]],
    residual: typing.Union[Form, Sequence[Form]],
    jacobian: typing.Union[Form, Sequence[Sequence[Form]]],
    bcs: Sequence[DirichletBC],
    _snes: PETSc.SNES,  # type: ignore[name-defined]
    x: PETSc.Vec,  # type: ignore[name-defined]
    b: PETSc.Vec,  # type: ignore[name-defined]
):
    """Assemble the residual at ``x`` into the vector ``b``.

    A function conforming to the interface expected by ``SNES.setFunction``
    can be created by fixing the first four arguments, e.g.:

    Example::

        snes = PETSc.SNES().create(mesh.comm)
        assemble_residual = functools.partial(
            dolfinx.fem.petsc.assemble_residual,
            u, residual, jacobian, bcs)
        snes.setFunction(assemble_residual, x, b)

    Args:
        u: Function(s) tied to the solution vector within the residual and
           Jacobian.
        residual: Form of the residual. It can be a sequence of forms.
        jacobian: Form of the Jacobian. It can be a nested sequence of
            forms.
        bcs: List of Dirichlet boundary conditions to lift the residual.
        _snes: The solver instance.
        x: The vector containing the point to evaluate the residual at.
        b: Vector to assemble the residual into.
    """
    # Update input vector before assigning
    dolfinx.la.petsc._ghost_update(x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore[attr-defined]

    # Assign the input vector to the unknowns
    assign(x, u)

    # Assign block data if block assembly is requested
    if isinstance(residual, Sequence) and b.getType() != PETSc.Vec.Type.NEST:  # type: ignore[attr-defined]
        _assign_block_data(residual, b)
        _assign_block_data(residual, x)

    # Assemble the residual
    dolfinx.la.petsc._zero_vector(b)
    assemble_vector(b, residual)  # type: ignore[arg-type]

    # Lift vector
    if isinstance(jacobian, Sequence):
        # Nest and blocked lifting
        bcs1 = _bcs_by_block(_extract_function_spaces(jacobian, 1), bcs)
        apply_lifting(b, jacobian, bcs=bcs1, x0=x, alpha=-1.0)
        dolfinx.la.petsc._ghost_update(b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
        bcs0 = _bcs_by_block(_extract_function_spaces(residual), bcs)  # type: ignore[arg-type]
        set_bc(b, bcs0, x0=x, alpha=-1.0)
    else:
        # Single form lifting
        apply_lifting(b, [jacobian], bcs=[bcs], x0=[x], alpha=-1.0)
        dolfinx.la.petsc._ghost_update(b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
        set_bc(b, bcs, x0=x, alpha=-1.0)
    dolfinx.la.petsc._ghost_update(b, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore[attr-defined]


def assemble_jacobian(
    u: typing.Union[Sequence[_Function], _Function],
    jacobian: typing.Union[Form, Sequence[Sequence[Form]]],
    preconditioner: typing.Optional[typing.Union[Form, Sequence[Sequence[Form]]]],
    bcs: Sequence[DirichletBC],
    _snes: PETSc.SNES,  # type: ignore[name-defined]
    x: PETSc.Vec,  # type: ignore[name-defined]
    J: PETSc.Mat,  # type: ignore[name-defined]
    P_mat: PETSc.Mat,  # type: ignore[name-defined]
):
    """Assemble the Jacobian and preconditioner matrices at ``x``
    into ``J`` and ``P_mat``.

    A function conforming to the interface expected by ``SNES.setJacobian``
    can be created by fixing the first four arguments e.g.:

    Example::

        snes = PETSc.SNES().create(mesh.comm)
        assemble_jacobian = functools.partial(
            dolfinx.fem.petsc.assemble_jacobian,
            u, jacobian, preconditioner, bcs)
        snes.setJacobian(assemble_jacobian, A, P_mat)

    Args:
        u: Function tied to the solution vector within the residual and
            jacobian.
        jacobian: Compiled form of the Jacobian.
        preconditioner: Compiled form of the preconditioner.
        bcs: List of Dirichlet boundary conditions to apply to the Jacobian
             and preconditioner matrices.
        _snes: The solver instance.
        x: The vector containing the point to evaluate at.
        J: Matrix to assemble the Jacobian into.
        P_mat: Matrix to assemble the preconditioner into.
    """
    # Copy existing soultion into the function used in the residual and
    # Jacobian
    dolfinx.la.petsc._ghost_update(x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore[attr-defined]
    assign(x, u)

    # Assemble Jacobian
    J.zeroEntries()
    assemble_matrix(J, jacobian, bcs, diag=1.0)  # type: ignore[arg-type, misc]
    J.assemble()
    if preconditioner is not None:
        P_mat.zeroEntries()
        assemble_matrix(P_mat, preconditioner, bcs, diag=1.0)  # type: ignore[arg-type, misc]
        P_mat.assemble()


class NonlinearProblem:
    """High-level class for solving nonlinear variational problems
    with PETSc SNES.

    Solves problems of the form
    :math:`F_i(u, v) = 0, i=0,\\ldots,N\\ \\forall v \\in V` where
    :math:`u=(u_0,\\ldots,u_N), v=(v_0,\\ldots,v_N)` using PETSc
    SNES as the non-linear solver.

    Note:
        The deprecated version of this class for use with
        :class:`dolfinx.nls.petsc.NewtonSolver` has been renamed
        :class:`dolfinx.fem.petsc.NewtonSolverNonlinearProblem`.

    Note:
        This high-level class automatically handles PETSc memory
        management. The user does not need to manually call
        ``.destroy()`` on returned PETSc objects.
    """

    def __init__(
        self,
        F: typing.Union[ufl.form.Form, Sequence[ufl.form.Form]],
        u: typing.Union[_Function, Sequence[_Function]],
        *,
        petsc_options_prefix: str,
        bcs: typing.Optional[Sequence[DirichletBC]] = None,
        J: typing.Optional[typing.Union[ufl.form.Form, Sequence[Sequence[ufl.form.Form]]]] = None,
        P: typing.Optional[typing.Union[ufl.form.Form, Sequence[Sequence[ufl.form.Form]]]] = None,
        kind: typing.Optional[typing.Union[str, Sequence[Sequence[str]]]] = None,
        petsc_options: typing.Optional[dict] = None,
        form_compiler_options: typing.Optional[dict] = None,
        jit_options: typing.Optional[dict] = None,
        entity_maps: typing.Optional[Sequence[_EntityMap]] = None,
    ):
        """
        Initialize solver for a nonlinear variational problem.

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
            bcs: Dirichlet boundary conditions.
            J: UFL form(s) representing the Jacobian
                :math:`J_{ij} = dF_i/du_j`. If not passed, derived
                automatically.
            P: UFL form(s) representing the preconditioner.
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
                as the integration domain, ``entity_maps`` must be
                supplied. For each key (a mesh, different to the
                integration domain mesh) a map should be provided relating
                the entities in the integration domain mesh to the entities
                in the key mesh e.g. for a key-value pair ``(msh, emap)``
                in ``entity_maps``, ``emap[i]`` is the entity in ``msh``
                corresponding to entity ``i`` in the integration domain
                mesh.
        """
        # Compile residual and Jacobian forms
        self._F = _create_form(
            F,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )

        if J is None:
            J = derivative_block(F, u)

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

        self._u = u
        # Set default values if not supplied
        bcs = [] if bcs is None else bcs

        # Create PETSc structures for the residual, Jacobian and solution
        # vector
        self._A = create_matrix(self.J, kind=kind)
        # Create PETSc structure for preconditioner if provided
        if self._preconditioner is not None:
            self._P_mat = create_matrix(self._preconditioner, kind=kind)
        else:
            self._P_mat = None

        # Determine the vector kind based on the matrix type
        kind = "nest" if self._A.getType() == PETSc.Mat.Type.NEST else kind  # type: ignore[attr-defined]
        assert kind is None or isinstance(kind, str)
        self._b = create_vector(self.F, kind=kind)
        self._x = create_vector(self.F, kind=kind)

        # Create the SNES solver and attach the corresponding Jacobian and
        # residual computation functions
        self._snes = PETSc.SNES().create(self.A.comm)  # type: ignore[attr-defined]
        self.solver.setJacobian(
            partial(assemble_jacobian, u, self.J, self.preconditioner, bcs), self.A, self.P_mat
        )
        self.solver.setFunction(partial(assemble_residual, u, self.F, self.J, bcs), self.b)

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
            opts = PETSc.Options()  # type: ignore[attr-defined]
            opts.prefixPush(self.solver.getOptionsPrefix())

            for k, v in petsc_options.items():
                opts[k] = v

            self.solver.setFromOptions()

            # Tidy up global options
            for k in petsc_options.keys():
                del opts[k]

            opts.prefixPop()

        if self.P_mat is not None and kind == "nest":
            # Transfer nest IS on self.P_mat to PC of main KSP. This allows
            # fieldsplit preconditioning to be applied, if desired.
            nest_IS = self.P_mat.getNestISs()
            fieldsplit_IS = tuple(
                [
                    (f"{u.name + '_' if u.name != 'f' else ''}{i}", IS)
                    for i, (u, IS) in enumerate(zip(self.u, nest_IS[0]))
                ]
            )
            self.solver.getKSP().getPC().setFieldSplitIS(*fieldsplit_IS)

    def solve(self) -> typing.Union[_Function, Sequence[_Function]]:
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
        dolfinx.la.petsc._ghost_update(self.x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore[attr-defined]

        # Copy solution back to function
        assign(self.x, self.u)

        return self.u

    def __del__(self):
        self._snes.destroy()
        self._x.destroy()
        self._A.destroy()
        self._b.destroy()
        if self._P_mat is not None:
            self._P_mat.destroy()

    @property
    def F(self) -> typing.Union[Form, Sequence[Form]]:
        """The compiled residual."""
        return self._F

    @property
    def J(self) -> typing.Union[Form, Sequence[Sequence[Form]]]:
        """The compiled Jacobian."""
        return self._J

    @property
    def preconditioner(self) -> typing.Optional[typing.Union[Form, Sequence[Sequence[Form]]]]:
        """The compiled preconditioner."""
        return self._preconditioner

    @property
    def A(self) -> PETSc.Mat:  # type: ignore[name-defined]
        """Jacobian matrix."""
        return self._A

    @property
    def P_mat(self) -> typing.Optional[PETSc.Mat]:  # type: ignore[name-defined]
        """Preconditioner matrix."""
        return self._P_mat

    @property
    def b(self) -> PETSc.Vec:  # type: ignore[name-defined]
        """Residual vector."""
        return self._b

    @property
    def x(self) -> PETSc.Vec:  # type: ignore[name-defined]
        """Solution vector.

        Note:
            The vector does not share memory with the
            solution function(s) ``u``.
        """
        return self._x

    @property
    def solver(self) -> PETSc.SNES:  # type: ignore[name-defined]
        """The SNES solver."""
        return self._snes

    @property
    def u(self) -> typing.Union[_Function, Sequence[_Function]]:
        """Solution function(s).

        Note:
            The function(s) do not share memory with the solution
            vector ``x``.
        """
        return self._u


# -- Deprecated non-linear problem class for NewtonSolver -----------------


class NewtonSolverNonlinearProblem:
    """(Deprecated) Nonlinear problem class for solving nonlinear
    problems using :class:`dolfinx.nls.petsc.NewtonSolver`.

    Solves problems of the form :math:`F(u, v) = 0 \\ \\forall v \\in V`
    using PETSc as the linear algebra backend.

    Note:
        This class is deprecated in favour of
        :class:`dolfinx.fem.petsc.NonlinearProblem`, a high level
        interface to SNES.

    Note:
        This class was previously called
        ``dolfinx.fem.petsc.NonlinearProblem``.
    """

    def __init__(
        self,
        F: ufl.form.Form,
        u: _Function,
        bcs: typing.Optional[Sequence[DirichletBC]] = None,
        J: ufl.form.Form = None,
        form_compiler_options: typing.Optional[dict] = None,
        jit_options: typing.Optional[dict] = None,
    ):
        """Initialize solver for solving a non-linear problem using
        Newton's method`.

        Args:
            F: The PDE residual F(u, v).
            u: The unknown.
            bcs: List of Dirichlet boundary conditions.
            J: UFL representation of the Jacobian (optional)
            form_compiler_options: Options used in FFCx
                compilation of this form. Run ``ffcx --help`` at the
                command line to see all available options.
            jit_options: Options used in CFFI JIT compilation of C
                code generated by FFCx. See ``python/dolfinx/jit.py``
                for all available options. Takes priority over all other
                option values.

        Example::

            problem = NonlinearProblem(F, u, [bc0, bc1])
        """
        warnings.warn(
            (
                "dolfinx.nls.petsc.NewtonSolver is deprecated. "
                + "Use dolfinx.fem.petsc.NonlinearProblem, "
                + "a high level interface to PETSc SNES, instead."
            ),
            DeprecationWarning,
        )

        self._L = _create_form(
            F, form_compiler_options=form_compiler_options, jit_options=jit_options
        )

        # Create the Jacobian matrix, dF/du
        if J is None:
            V = u.function_space
            du = ufl.TrialFunction(V)
            J = ufl.derivative(F, u, du)

        self._a = _create_form(
            J, form_compiler_options=form_compiler_options, jit_options=jit_options
        )
        self.bcs = bcs

    @property
    def L(self) -> Form:
        """The compiled linear form (the residual form)."""
        return self._L

    @property
    def a(self) -> Form:
        """The compiled bilinear form (the Jacobian form)."""
        return self._a

    def form(self, x: PETSc.Vec) -> None:  # type: ignore[name-defined]
        """This function is called before the residual or Jacobian is
        computed. This is usually used to update ghost values.

        Args:
           x: The vector containing the latest solution
        """
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore[attr-defined]

    def F(self, x: PETSc.Vec, b: PETSc.Vec) -> None:  # type: ignore[name-defined]
        """Assemble the residual F into the vector b.

        Args:
            x: The vector containing the latest solution
            b: Vector to assemble the residual into
        """
        # Reset the residual vector
        dolfinx.la.petsc._zero_vector(b)
        assemble_vector(b, self._L)

        # Apply boundary condition
        if self.bcs is not None:
            apply_lifting(b, [self._a], bcs=[self.bcs], x0=[x], alpha=-1.0)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
            set_bc(b, self.bcs, x, -1.0)
        else:
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]

    def J(self, x: PETSc.Vec, A: PETSc.Mat) -> None:  # type: ignore[name-defined]
        """Assemble the Jacobian matrix.

        Args:
            x: The vector containing the latest solution
        """
        A.zeroEntries()
        assemble_matrix(A, self._a, self.bcs)  # type: ignore[arg-type]
        A.assemble()


# -- Additional free helper functions (interpolations, assignments etc.) --


def discrete_curl(space0: _FunctionSpace, space1: _FunctionSpace) -> PETSc.Mat:  # type: ignore[name-defined]
    """Assemble a discrete curl operator.

    Args:
        space0: H1 space to interpolate the gradient from.
        space1: H(curl) space to interpolate into.

    Returns:
        Discrete curl operator.
    """
    return _discrete_curl(space0._cpp_object, space1._cpp_object)


def discrete_gradient(space0: _FunctionSpace, space1: _FunctionSpace) -> PETSc.Mat:  # type: ignore[name-defined]
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
    return _discrete_gradient(space0._cpp_object, space1._cpp_object)


def interpolation_matrix(V0: _FunctionSpace, V1: _FunctionSpace) -> PETSc.Mat:  # type: ignore[name-defined]
    r"""Assemble an interpolation operator matrix for discreye
    interpolation between finite element spaces.

    Consider is the vector of degrees-of-freedom  :math:`u_{i}`
    associated with a function in :math:`V_{i}`. This function returns
    the matrix :math:`\Pi` sucht that

    .. math::

        u_{1} = \Pi u_{0}.

    Args:
        V0: Space to interpolate from.
        V1: Space to interpolate into.

    Returns:
        The interpolation matrix :math:`\Pi`.
    """
    return _interpolation_matrix(V0._cpp_object, V1._cpp_object)


@functools.singledispatch
def assign(u: typing.Union[_Function, Sequence[_Function]], x: PETSc.Vec):  # type: ignore[name-defined]
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
    if x.getType() == PETSc.Vec.Type().NEST:  # type: ignore[attr-defined]
        dolfinx.la.petsc.assign([v.x.array for v in u], x)
    else:
        if isinstance(u, Sequence):
            data0, data1 = [], []
            for v in u:
                bs = v.function_space.dofmap.bs
                n = v.function_space.dofmap.index_map.size_local
                data0.append(v.x.array[: bs * n])
                data1.append(v.x.array[bs * n :])
            dolfinx.la.petsc.assign(data0 + data1, x)
        else:
            dolfinx.la.petsc.assign(u.x.array, x)


@assign.register
def _(x: PETSc.Vec, u: typing.Union[_Function, Sequence[_Function]]):  # type: ignore[name-defined]
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
    if x.getType() == PETSc.Vec.Type().NEST:  # type: ignore[attr-defined]
        dolfinx.la.petsc.assign(x, [v.x.array for v in u])
    else:
        if isinstance(u, Sequence):
            data0, data1 = [], []
            for v in u:
                bs = v.function_space.dofmap.bs
                n = v.function_space.dofmap.index_map.size_local
                data0.append(v.x.array[: bs * n])
                data1.append(v.x.array[bs * n :])
            dolfinx.la.petsc.assign(x, data0 + data1)
        else:
            dolfinx.la.petsc.assign(x, u.x.array)
