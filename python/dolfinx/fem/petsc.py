# Copyright (C) 2018-2025 Garth N. Wells and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Assembly functions into PETSc objects for variational forms.

Functions in this module generally apply functions in :mod:`dolfinx.fem`
to PETSc linear algebra objects and handle any PETSc-specific
preparation.

Note:
    Due to subtle issues in the interaction between petsc4py memory
    management and the Python garbage collector, it is recommended that
    the PETSc method ``destroy()`` is called on returned PETSc objects
    once the object is no longer required. Note that ``destroy()`` is
    collective over the object's MPI communicator.
"""

# mypy: ignore-errors

from __future__ import annotations

import contextlib
import functools
import itertools
import typing
from collections.abc import Iterable, Sequence

from petsc4py import PETSc

# ruff: noqa: E402
import dolfinx

assert dolfinx.has_petsc4py

import numpy as np
from numpy import typing as npt

import dolfinx.cpp as _cpp
import dolfinx.la.petsc
import ufl
from dolfinx.cpp.fem.petsc import discrete_curl as _discrete_curl
from dolfinx.cpp.fem.petsc import discrete_gradient as _discrete_gradient
from dolfinx.cpp.fem.petsc import interpolation_matrix as _interpolation_matrix
from dolfinx.fem import IntegralType, pack_coefficients, pack_constants
from dolfinx.fem.assemble import apply_lifting as _apply_lifting
from dolfinx.fem.assemble import assemble_vector as _assemble_vector
from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.bcs import bcs_by_block as _bcs_by_block
from dolfinx.fem.forms import Form
from dolfinx.fem.forms import extract_function_spaces as _extract_spaces
from dolfinx.fem.forms import form as _create_form
from dolfinx.fem.function import Function as _Function
from dolfinx.fem.function import FunctionSpace as _FunctionSpace

__all__ = [
    "LinearProblem",
    "NonlinearProblem",
    "apply_lifting",
    "assemble_matrix",
    "assemble_vector",
    "assign",
    "create_matrix",
    "create_vector",
    "discrete_curl",
    "discrete_gradient",
    "interpolation_matrix",
    "set_bc",
]


def _extract_function_spaces(
    a: Iterable[Iterable[Form]],
) -> tuple[list[_FunctionSpace], list[_FunctionSpace]]:
    """From a rectangular array of bilinear forms, extract the function
    spaces for each block row and block column.
    """
    assert len({len(cols) for cols in a}) == 1, "Array of function spaces is not rectangular"

    # Extract (V0, V1) pair for each block in 'a'
    def fn(form):
        return form.function_spaces if form is not None else None

    from functools import partial

    Vblock: typing.Iterable = map(partial(map, fn), a)

    # Compute spaces for each row/column block
    rows: list[set] = [set() for i in range(len(a))]
    cols: list[set] = [set() for i in range(len(a[0]))]
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


# -- Vector instantiation ----------------------------------------------------


def create_vector(
    L: typing.Union[Form, Iterable[Form]], kind: typing.Optional[str] = None
) -> PETSc.Vec:
    """Create a PETSc vector that is compatible with a linear form(s).

    Three cases are supported:

    1. For a single linear form ``L``, if ``kind`` is ``None`` or is
       ``PETSc.Vec.Type.MPI``, a ghosted PETSc vector which is
       compatible with ``L`` is created.

    2. If ``L`` is a sequence of linear forms and ``kind`` is ``None``
       or is ``PETSc.Vec.Type.MPI``, a ghosted PETSc vector which is
       compatible with ``L`` is created. The created vector ``b`` is
       initialized such that on each MPI process ``b = [b_0, b_1, ...,
       b_n, b_0g, b_1g, ..., b_ng]``, where ``b_i`` are the entries
       associated with the 'owned' degrees-of-freedom for ``L[i]`` and
       ``b_ig`` are the 'unowned' (ghost) entries for ``L[i]``.

       For this case, the returned vector has an attribute ``_blocks``
       that holds the local offsets into ``b`` for the (i) owned and
       (ii) ghost entries for each ``L[i]``. It can be accessed by
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

    3. If ``L`` is a sequence of linear forms and ``kind`` is
       ``PETSc.Vec.Type.NEST``, a PETSc nested vector (a 'nest' of
       ghosted PETSc vectors) which is compatible with ``L`` is created.

    Args:
        L: Linear form or a sequence of linear forms.
        kind: PETSc vector type (``VecType``) to create.

    Returns:
        A PETSc vector with a layout that is compatible with ``L``. The
        vector is not initialised to zero.
    """
    try:
        dofmap = L.function_spaces[0].dofmaps(0)  # Single form case
        return dolfinx.la.petsc.create_vector(dofmap.index_map, dofmap.index_map_bs)
    except AttributeError:
        maps = [
            (
                form.function_spaces[0].dofmaps(0).index_map,
                form.function_spaces[0].dofmaps(0).index_map_bs,
            )
            for form in L
        ]
        if kind == PETSc.Vec.Type.NEST:
            return _cpp.fem.petsc.create_vector_nest(maps)
        elif kind == PETSc.Vec.Type.MPI or kind is None:
            off_owned = tuple(
                itertools.accumulate(maps, lambda off, m: off + m[0].size_local * m[1], initial=0)
            )
            off_ghost = tuple(
                itertools.accumulate(
                    maps, lambda off, m: off + m[0].num_ghosts * m[1], initial=off_owned[-1]
                )
            )

            b = _cpp.fem.petsc.create_vector_block(maps)
            b.setAttr("_blocks", (off_owned, off_ghost))
            return b
        else:
            raise NotImplementedError(
                "Vector type must be specified for blocked/nested assembly."
                f"Vector type '{kind}' not supported."
                "Did you mean 'nest' or 'mpi'?"
            )


# -- Matrix instantiation ----------------------------------------------------


def create_matrix(
    a: typing.Union[Form, Iterable[Iterable[Form]]],
    kind: typing.Optional[typing.Union[str, Iterable[Iterable[str]]]] = None,
) -> PETSc.Mat:
    """Create a PETSc matrix that is compatible with the (sequence) of bilinear form(s).

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
        a: A bilinear form or a nested list of bilinear forms.
        kind: The PETSc matrix type (``MatType``).

    Returns:
        A PETSc matrix.
    """
    try:
        return _cpp.fem.petsc.create_matrix(a._cpp_object, kind)  # Single form
    except AttributeError:  # ``a`` is a nested list
        _a = [[None if form is None else form._cpp_object for form in arow] for arow in a]
        if kind == PETSc.Mat.Type.NEST:  # Create nest matrix with default types
            return _cpp.fem.petsc.create_matrix_nest(_a, None)
        else:
            try:
                return _cpp.fem.petsc.create_matrix_block(_a, kind)  # Single 'kind' type
            except TypeError:
                return _cpp.fem.petsc.create_matrix_nest(_a, kind)  # Array of 'kind' types


# -- Vector assembly ---------------------------------------------------------


@functools.singledispatch
def assemble_vector(
    L: typing.Union[Form, Iterable[Form]],
    constants: typing.Optional[npt.NDArray, Iterable[npt.NDArray]] = None,
    coeffs: typing.Optional[
        typing.Union[
            dict[tuple[IntegralType, int], npt.NDArray],
            Iterable[dict[tuple[IntegralType, int], npt.NDArray]],
        ]
    ] = None,
    kind: typing.Optional[str] = None,
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
        L: A linear form or list of linear forms.
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
    if kind == PETSc.Vec.Type.NEST:
        for b_sub in b.getNestSubVecs():
            with b_sub.localForm() as b_local:
                b_local.set(0.0)
    else:
        with b.localForm() as b_local:
            b_local.set(0)
    return assemble_vector(b, L, constants, coeffs)


@assemble_vector.register(PETSc.Vec)
def _assemble_vector_vec(
    b: PETSc.Vec,
    L: typing.Union[Form, Iterable[Form]],
    constants: typing.Optional[npt.NDArray, Iterable[npt.NDArray]] = None,
    coeffs: typing.Optional[
        typing.Union[
            dict[tuple[IntegralType, int], npt.NDArray],
            Iterable[dict[tuple[IntegralType, int], npt.NDArray]],
        ]
    ] = None,
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
        L: A linear form or list of linear forms to assemble into ``b``.
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
        constants = [None] * len(L) if constants is None else constants
        coeffs = [None] * len(L) if coeffs is None else coeffs
        for b_sub, L_sub, const, coeff in zip(b.getNestSubVecs(), L, constants, coeffs):
            with b_sub.localForm() as b_local:
                _assemble_vector(b_local.array_w, L_sub, const, coeff)
    elif isinstance(L, Iterable):
        constants = pack_constants(L) if constants is None else constants
        coeffs = pack_coefficients(L) if coeffs is None else coeffs
        offset0, offset1 = b.getAttr("_blocks")
        with b.localForm() as b_l:
            for L_, const, coeff, off0, off1, offg0, offg1 in zip(
                L, constants, coeffs, offset0, offset0[1:], offset1, offset1[1:]
            ):
                bx_ = np.zeros((off1 - off0) + (offg1 - offg0), dtype=PETSc.ScalarType)
                _assemble_vector(bx_, L_, const, coeff)
                size = off1 - off0
                b_l.array_w[off0:off1] += bx_[:size]
                b_l.array_w[offg0:offg1] += bx_[size:]
    else:
        with b.localForm() as b_local:
            _assemble_vector(b_local.array_w, L, constants, coeffs)

    return b


# -- Matrix assembly ---------------------------------------------------------
@functools.singledispatch
def assemble_matrix(
    a: typing.Union[Form, Iterable[Iterable[Form]]],
    bcs: typing.Optional[Iterable[DirichletBC]] = None,
    diag: float = 1,
    constants: typing.Optional[
        typing.Union[Iterable[np.ndarray], Iterable[Iterable[np.ndarray]]]
    ] = None,
    coeffs: typing.Optional[
        typing.Union[
            dict[tuple[IntegralType, int], npt.NDArray],
            Iterable[dict[tuple[IntegralType, int], npt.NDArray]],
        ]
    ] = None,
    kind=None,
):
    """Assemble a bilinear form into a matrix.

    The following cases are supported:

    1. If ``a`` is a single bilinear form, the form is assembled
       into PETSc matrix of type ``kind``.
    #. If ``a`` is a :math:`m \times n` rectangular array of forms the
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
    try:
        A = _cpp.fem.petsc.create_matrix(a._cpp_object, kind)
    except AttributeError:
        A = create_matrix(a, kind)
    assemble_matrix(A, a, bcs, diag, constants, coeffs)
    return A


def _assemble_matrix_block_mat(
    A: PETSc.Mat,
    a: Iterable[Iterable[Form]],
    bcs: typing.Optional[Iterable[DirichletBC]],
    diag: float,
    constants: typing.Optional[Iterable[npt.NDArray]],
    coeffs: typing.Optional[Iterable[Iterable[dict[tuple[IntegralType, int], npt.NDArray]]]],
) -> PETSc.Mat:
    """Assemble bilinear forms into a blocked matrix."""
    consts = [pack_constants(forms) for forms in a] if constants is None else constants
    coeffs = [pack_coefficients(forms) for forms in a] if coeffs is None else coeffs
    V = _extract_function_spaces(a)
    is0 = _cpp.la.petsc.create_index_sets(
        [(Vsub.dofmaps(0).index_map, Vsub.dofmaps(0).index_map_bs) for Vsub in V[0]]
    )
    is1 = _cpp.la.petsc.create_index_sets(
        [(Vsub.dofmaps(0).index_map, Vsub.dofmaps(0).index_map_bs) for Vsub in V[1]]
    )

    _bcs = [bc._cpp_object for bc in bcs] if bcs is not None else []

    bcs0 = _bcs_by_block(_extract_spaces(a, 0), bcs) if bcs is not None else []
    bcs1 = _bcs_by_block(_extract_spaces(a, 1), bcs) if bcs is not None else []
    # print("**** Test0:", bcs0)
    # print("**** Test1:", bcs1)

    for i, a_row in enumerate(a):
        for j, a_sub in enumerate(a_row):
            if a_sub is not None:
                Asub = A.getLocalSubMatrix(is0[i], is1[j])
                V0, V1 = a_sub.function_spaces[0], a_sub.function_spaces[0]
                bcs0 = (
                    [bc._cpp_object for bc in bcs if V0.contains(bc.function_space)]
                    if bcs is not None
                    else []
                )
                bcs1 = (
                    [bc._cpp_object for bc in bcs if V1.contains(bc.function_space)]
                    if bcs is not None
                    else []
                )

                _cpp.fem.petsc.assemble_matrix(
                    Asub, a_sub._cpp_object, consts[i][j], coeffs[i][j], bcs0, bcs1, True
                )
                A.restoreLocalSubMatrix(is0[i], is1[j], Asub)
            elif i == j:
                print("\nTest:", bcs0[i], bcs1[i])
                # row_forms = [row_form for row_form in a_row if row_form is not None]
                # assert len(row_forms) > 0
                # for bc in _bcs:
                #     if row_forms[0].function_spaces[0].contains(bc.function_space):
                #         raise RuntimeError(
                #             f"Diagonal sub-block ({i}, {j}) cannot be 'None' "
                #             " and have DirichletBC applied."
                #             " Consider assembling a zero block."
                #         )

    # Flush to enable switch from add to set in the matrix
    A.assemble(PETSc.Mat.AssemblyType.FLUSH)

    # Set diagonal
    for i, a_row in enumerate(a):
        for j, a_sub in enumerate(a_row):
            if a_sub is not None:
                Asub = A.getLocalSubMatrix(is0[i], is1[j])
                if a_sub.function_spaces[0] is a_sub.function_spaces[1]:
                    _cpp.fem.petsc.insert_diagonal(Asub, a_sub.function_spaces[0], _bcs, diag)
                A.restoreLocalSubMatrix(is0[i], is1[j], Asub)

    return A


@assemble_matrix.register
def assemble_matrix_mat(
    A: PETSc.Mat,
    a: typing.Union[Form, Iterable[Iterable[Form]]],
    bcs: typing.Optional[Iterable[DirichletBC]] = None,
    diag: float = 1,
    constants: typing.Optional[typing.Union[np.ndarray, Iterable[Iterable[np.ndarray]]]] = None,
    coeffs: typing.Optional[
        typing.Union[
            dict[tuple[IntegralType, int], npt.NDArray],
            Iterable[Iterable[dict[tuple[IntegralType, int], npt.NDArray]]],
        ]
    ] = None,
) -> PETSc.Mat:
    """Assemble bilinear form into a matrix.

    The matrix vector ``A`` must have been initialized with a
    size/layout that is consistent with the bilinear form(s). The PETSc
    matrix ``A`` is normally created by :func:`create_matrix`.bcs

    The returned matrix is not finalised, i.e. ghost values are not
    accumulated.
    """

    if A.getType() == PETSc.Mat.Type.NEST:
        bcs0 = _bcs_by_block(_extract_spaces(a, 0), bcs) if bcs is not None else []
        bcs1 = _bcs_by_block(_extract_spaces(a, 1), bcs) if bcs is not None else []
        constants = [pack_constants(forms) for forms in a] if constants is None else constants
        coeffs = [pack_coefficients(forms) for forms in a] if coeffs is None else coeffs
        for i, (a_row, const_row, coeff_row) in enumerate(zip(a, constants, coeffs)):
            for j, (a_block, const, coeff) in enumerate(zip(a_row, const_row, coeff_row)):
                if a_block is not None:
                    Asub = A.getNestSubMatrix(i, j)
                    assemble_matrix(Asub, a_block, bcs, diag, const, coeff)
                elif i == j:
                    print("\nTestX:", bcs0[i], bcs1[i])
                    row_forms = [row_form for row_form in a_row if row_form is not None]
                    assert len(row_forms) > 0
                    for bc in bcs:
                        if row_forms[0].function_spaces[0].contains(bc.function_space):
                            raise RuntimeError(
                                f"Diagonal sub-block ({i}, {j}) cannot be 'None'"
                                " and have DirichletBC applied."
                                " Consider assembling a zero block."
                            )
    elif isinstance(a, Iterable):  # Blocked
        _assemble_matrix_block_mat(A, a, bcs, diag, constants, coeffs)
    else:  # Non-blocked
        print("A0000")
        constants = pack_constants(a) if constants is None else constants
        coeffs = pack_coefficients(a) if coeffs is None else coeffs
        V0, V1 = a.function_spaces[0], a.function_spaces[0]
        bcs0 = (
            [bc._cpp_object for bc in bcs if V0.contains(bc.function_space)]
            if bcs is not None
            else []
        )
        bcs1 = (
            [bc._cpp_object for bc in bcs if V1.contains(bc.function_space)]
            if bcs is not None
            else []
        )
        print("bc", bcs0, bcs1)
        _cpp.fem.petsc.assemble_matrix(A, a._cpp_object, constants, coeffs, bcs0, bcs1)
        print("bc-posy")
        if V0 is V1:
            A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
            A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
            _bcs = [bc._cpp_object for bc in bcs] if bcs is not None else []
            _cpp.fem.petsc.insert_diagonal(A, V0, _bcs, diag)
        print("B0000")

    return A


# -- Modifiers for Dirichlet conditions ---------------------------------------


def apply_lifting(
    b: PETSc.Vec,
    a: typing.Union[Form, Iterable[Form], Iterable[Iterable[Form]]],
    bcs: typing.Optional[typing.Union[Iterable[DirichletBC], Iterable[Iterable[DirichletBC]]]],
    x0: typing.Optional[Iterable[PETSc.Vec]] = None,
    alpha: float = 1,
    constants: typing.Optional[
        typing.Union[Iterable[np.ndarray], Iterable[Iterable[np.ndarray]]]
    ] = None,
    coeffs=None,
) -> None:
    """Modify the right-hand side PETSc vector ``b`` to account for
    constraints (Dirichlet boundary conitions).

    See :func:`dolfinx.fem.apply_lifting` for a mathematical
    descriptions of the lifting operation.

    Args:
        b: Vector to modify in-place.
        a: List of bilinear forms. If ``b`` is not blocked or a nest,
            then ``a`` is a 1D sequence. If ``b`` is blocked or a nest,
            then ``a`` is  a 2D array of forms, with the ``a[i]`` forms
            used to modify the block/nest vector ``b[i]``.
        bcs: Boundary conditions used to modify ``b`` (see
            :func:`dolfinx.fem.apply_lifting`). Two cases are supported:

            1. The boundary conditions ``bcs`` is a 'list-of-lists' such
               that ``bcs[j]`` are the Dirichlet boundary conditionns
               associated with the forms in the ``j`` th colulmn of
               ``a``. Helper functions exist to create a list-of-lists
               of `DirichletBC` from the 2D ``a`` and a flat list of
               `DirichletBC` objects ``bcs``::

                   bcs1 = fem.bcs_by_block(fem.extract_function_spaces(a, 1), bcs)

            2. ``bcs`` is a sequence of :class:`dolfinx.fem.DirichletBC`
               objects. The function deduces which `DiricletBC` objects
               apply to each column of ``a`` by matching the
               :class:`dolfinx.fem.FunctionSpace`.

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
    if b.getType() == PETSc.Vec.Type.NEST:
        try:
            bcs = _bcs_by_block(_extract_spaces(a, 1), bcs)
        except AttributeError:
            pass
        x0 = [] if x0 is None else x0.getNestSubVecs()
        constants = [pack_constants(forms) for forms in a] if constants is None else constants
        coeffs = [pack_coefficients(forms) for forms in a] if coeffs is None else coeffs
        for b_sub, a_sub, const, coeff in zip(b.getNestSubVecs(), a, constants, coeffs):
            const_ = list(
                map(lambda c: np.array([], dtype=PETSc.ScalarType) if c is None else c, const)
            )
            apply_lifting(b_sub, a_sub, bcs, x0, alpha, const_, coeff)
    else:
        with contextlib.ExitStack() as stack:
            if b.getAttr("_blocks") is not None:
                if x0 is not None:
                    offset0, offset1 = x0.getAttr("_blocks")
                    xl = stack.enter_context(x0.localForm())
                    xlocal = [
                        np.concat((xl[off0:off1], xl[offg0:offg1]))
                        for (off0, off1, offg0, offg1) in zip(
                            offset0, offset0[1:], offset1, offset1[1:]
                        )
                    ]
                else:
                    xlocal = None

                try:
                    bcs = _bcs_by_block(_extract_spaces(a, 1), bcs)
                except AttributeError:
                    pass
                offset0, offset1 = b.getAttr("_blocks")
                with b.localForm() as b_l:
                    for i, (a_, off0, off1, offg0, offg1) in enumerate(
                        zip(a, offset0, offset0[1:], offset1, offset1[1:])
                    ):
                        const = pack_constants(a_) if constants is None else constants[i]
                        coeff = pack_coefficients(a_) if coeffs is None else coeffs[i]
                        const_ = [
                            np.empty(0, dtype=PETSc.ScalarType) if val is None else val
                            for val in const
                        ]
                        bx_ = np.concat((b_l[off0:off1], b_l[offg0:offg1]))
                        _apply_lifting(bx_, a_, bcs, xlocal, float(alpha), const_, coeff)
                        size = off1 - off0
                        b_l.array_w[off0:off1] = bx_[:size]
                        b_l.array_w[offg0:offg1] = bx_[size:]
            else:
                x0 = [] if x0 is None else x0
                x0 = [stack.enter_context(x.localForm()) for x in x0]
                x0_r = [x.array_r for x in x0]
                b_local = stack.enter_context(b.localForm())
                try:
                    # print("\n a:", a)
                    # print("\n bs:", bcs)
                    bcs = _bcs_by_block(_extract_spaces(a, 0), bcs)
                except AttributeError:
                    pass
                try:
                    _apply_lifting(b_local.array_w, a, bcs, x0_r, alpha, constants, coeffs)
                except TypeError:
                    _apply_lifting(b_local.array_w, [a], bcs, x0_r, alpha, constants, coeffs)
    return b


def set_bc(
    b: PETSc.Vec,
    bcs: typing.Union[Iterable[DirichletBC], Iterable[Iterable[DirichletBC]]],
    x0: typing.Optional[PETSc.Vec] = None,
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
    if b.getType() == PETSc.Vec.Type.NEST:
        _b = b.getNestSubVecs()
        x0 = len(_b) * [None] if x0 is None else x0.getNestSubVecs()
        for b_sub, bc, x_sub in zip(_b, bcs, x0):
            set_bc(b_sub, bc, x_sub, alpha)
    else:
        try:
            offset0, _ = b.getAttr("_blocks")
            b_array = b.getArray(readonly=False)
            x_array = x0.getArray(readonly=True) if x0 is not None else None
            for bcs0, off0, off1 in zip(bcs, offset0, offset0[1:]):
                x0_sub = x_array[off0:off1] if x0 is not None else None
                for bc in bcs0:
                    bc.set(b_array[off0:off1], x0_sub, alpha)
        except TypeError:
            x0 = x0.array_r if x0 is not None else None
            try:
                for bc in bcs:
                    bc.set(b.array_w, x0, alpha)
            except AttributeError:
                assert len(bcs) == 1
                for bc in bcs[0]:
                    bc.set(b.array_w, x0, alpha)


class LinearProblem:
    """Class for solving a linear variational problem.

    Represents a problem of the form: find :math:`u \\in V` such that

    .. math::

        a(u, v) = L(v) \\forall v \\in W.

    where ``a`` is a bilinear form, ``L`` is a linear form.

    PETSc is used as a thelinear algebra backend.
    """

    def __init__(
        self,
        a: typing.Union[ufl.Form, Iterable[Iterable[ufl.Form]]],
        L: typing.Union[ufl.Form, Iterable[ufl.Form]],
        bcs: typing.Optional[Iterable[DirichletBC]] = None,
        u: typing.Optional[typing.Union[_Function, Iterable[_Function]]] = None,
        petsc_options: typing.Optional[dict] = None,
        form_compiler_options: typing.Optional[dict] = None,
        jit_options: typing.Optional[dict] = None,
        kind: typing.Optional[typing.Union[str, Iterable[Iterable[str]]]] = None,
    ) -> None:
        """Initialize solver for a linear variational problem.

        Args:
            a: Bilinear form or a rectangular array of bilinear
                forms.
            L: Linear form or a sequence of lienar forms.
            bcs: Sequence of Dirichlet boundary conditions.
            u: Solution function. It is be created if not provided.
            petsc_options: Options that are passed to the PETSc linear
                solver. For available opions see the `PETSc
                documentation
                <https://petsc4py.readthedocs.io/en/stable/manual/ksp/>`_.
            form_compiler_options: Options used in FFCx compilation of
                this form. Run ``ffcx --help`` at the commandline to see
                available options.
            jit_options: Options used in CFFI JIT compilation of C
                code generated by FFCx. See ``python/dolfinx/jit.py``
                for all available options. Takes priority over all other
                option values.
            kind: The PETSc matrix. See :func:`create_matrix` for
                options.

        Example::

            problem = LinearProblem(a, L, [bc0, bc1], petsc_options={"ksp_type": "preonly",
                                                                     "pc_type": "lu",
                                                                     "pc_factor_mat_solver_type":
                                                                       "mumps"})

            problem = LinearProblem([[a00,a01],[None, a11]], [L0, L1], bcs=[bc0, bc1],
                                    u=[uh0, uh1])
        """
        self._a = _create_form(
            a,
            dtype=PETSc.ScalarType,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
        )
        self._L = _create_form(
            L,
            dtype=PETSc.ScalarType,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
        )

        self._A = create_matrix(self._a, kind=kind)
        kind = "nest" if self._A.getType() == PETSc.Mat.Type.NEST else kind
        self._b = create_vector(self._L, kind=kind)
        self._x = create_vector(self._L, kind=kind)
        if u is None:
            # Extract function space(s) for u from L
            try:
                self._u = _Function(L.arguments()[0].ufl_function_space())
            except AttributeError:
                self._u = [_Function(_L.arguments()[0].ufl_function_space()) for _L in L]
        else:
            self._u = u

        self._bcs0 = _bcs_by_block(_extract_spaces(self._L), bcs) if bcs is not None else None
        self._bcs1 = _bcs_by_block(_extract_spaces(self._a, 1), bcs) if bcs is not None else None

        try:
            comm = self._u.function_space.mesh.comm
        except AttributeError:
            comm = self._u[0].function_space.mesh.comm

        self._solver = PETSc.KSP().create(comm)
        self._solver.setOperators(self._A)

        # Give PETSc solver options a unique prefix
        problem_prefix = f"dolfinx_solve_{id(self)}"
        self._solver.setOptionsPrefix(problem_prefix)

        # Set PETSc options on the solver
        opts = PETSc.Options()
        opts.prefixPush(problem_prefix)
        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v
        opts.prefixPop()
        self._solver.setFromOptions()

        # Set matrix and vector PETSc options
        self._A.setOptionsPrefix(problem_prefix)
        self._A.setFromOptions()
        self._b.setOptionsPrefix(problem_prefix)
        self._b.setFromOptions()

    def __del__(self):
        self._solver.destroy()
        self._A.destroy()
        self._b.destroy()
        self._x.destroy()

    def solve(self) -> typing.Union[_Function, Iterable[_Function]]:
        """Solve the problem."""

        # Assemble lhs
        # TODO: support bcs0, bcs1 in assemble_matrix
        bc0 = [bc for bcs in self._bcs0 for bc in bcs] if self._bcs0 is not None else []
        bc1 = [bc for bcs in self._bcs1 for bc in bcs] if self._bcs0 is not None else []
        bcs = list(set(bc0 + bc1))
        self._A.zeroEntries()
        assemble_matrix(self._A, self._a, bcs=bcs)
        self._A.assemble()

        # Zero b and assemble
        if self._b.getType() == PETSc.Vec.Type.NEST:
            for b_sub in self._b.getNestSubVecs():
                with b_sub.localForm() as b_local:
                    b_local.set(0.0)
        else:
            with self._b.localForm() as b_loc:
                b_loc.set(0)
        assemble_vector(self._b, self._L)

        # Apply boundary conditions to the RHS
        if self._bcs1 is not None:
            apply_lifting(self._b, self._a, bcs=self._bcs1)
        dolfinx.la.petsc._ghost_update(self._b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        if self._bcs0 is not None:
            dolfinx.fem.petsc.set_bc(self._b, self._bcs0)

        # Solve linear system and update ghost values in the solution
        self._solver.solve(self._b, self._x)
        dolfinx.la.petsc._ghost_update(self._x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        dolfinx.fem.petsc.assign(self._x, self._u)
        return self._u

    @property
    def L(self) -> Form:
        """The compiled linear form."""
        return self._L

    @property
    def a(self) -> Form:
        """The compiled bilinear form."""
        return self._a

    @property
    def A(self) -> PETSc.Mat:
        """Matrix operator"""
        return self._A

    @property
    def b(self) -> PETSc.Vec:
        """Right-hand side vector"""
        return self._b

    @property
    def solver(self) -> PETSc.KSP:
        """Linear solver object"""
        return self._solver

    @property
    def u(self) -> typing.Union[_Function, list[_Function]]:
        """Solution function"""
        return self._u


class NonlinearProblem:
    """Nonlinear problem class for solving the non-linear problems.

    Solves problems of the form :math:`F(u, v) = 0 \\ \\forall v \\in V` using
    PETSc as the linear algebra backend.
    """

    def __init__(
        self,
        F: ufl.form.Form,
        u: _Function,
        bcs: typing.Optional[Iterable[DirichletBC]] = None,
        J: ufl.form.Form = None,
        form_compiler_options: typing.Optional[dict] = None,
        jit_options: typing.Optional[dict] = None,
    ):
        """Initialize solver for solving a non-linear problem using Newton's method`.

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

    def form(self, x: PETSc.Vec) -> None:
        """This function is called before the residual or Jacobian is
        computed. This is usually used to update ghost values.

        Args:
           x: The vector containing the latest solution
        """
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def F(self, x: PETSc.Vec, b: PETSc.Vec) -> None:
        """Assemble the residual F into the vector b.

        Args:
            x: The vector containing the latest solution
            b: Vector to assemble the residual into
        """
        # Reset the residual vector
        with b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(b, self._L)

        # Apply boundary condition
        if self.bcs is not None:
            apply_lifting(b, [self._a], bcs=[self.bcs], x0=[x], alpha=-1.0)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, self.bcs, x, -1.0)
        else:
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    def J(self, x: PETSc.Vec, A: PETSc.Mat) -> None:
        """Assemble the Jacobian matrix.

        Args:
            x: The vector containing the latest solution
        """
        A.zeroEntries()
        assemble_matrix(A, self._a, self.bcs)
        A.assemble()


def discrete_curl(space0: _FunctionSpace, space1: _FunctionSpace) -> PETSc.Mat:
    """Assemble a discrete curl operator.

    Args:
        space0: H1 space to interpolate the gradient from.
        space1: H(curl) space to interpolate into.

    Returns:
        Discrete curl operator.
    """
    return _discrete_curl(space0._cpp_object, space1._cpp_object)


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
    return _discrete_gradient(space0._cpp_object, space1._cpp_object)


def interpolation_matrix(V0: _FunctionSpace, V1: _FunctionSpace) -> PETSc.Mat:
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
def assign(u: typing.Union[_Function, Sequence[_Function]], x: PETSc.Vec):
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
    if x.getType() == PETSc.Vec.Type().NEST:
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


@assign.register(PETSc.Vec)
def _(x: PETSc.Vec, u: typing.Union[_Function, Sequence[_Function]]):
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
