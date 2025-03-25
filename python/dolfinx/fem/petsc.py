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
from dolfinx.fem import pack_coefficients, pack_constants
from dolfinx.fem.assemble import _assemble_vector_array
from dolfinx.fem.assemble import apply_lifting as _apply_lifting
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
    "assemble_matrix_block",
    "assemble_vector",
    "assemble_vector_block",
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

    If the vector type is not specified (``kind=None``) or is
    ``PETSc.Vec.Type.MPI``, a ghosted PETSc vector which is compatible
    with ``L`` is created. If the vector type is
    ``PETSc.Vec.Type.NEST``, a PETSc nested vector (a nest of ghosted
    PETSc vectors) which is compatible with ``L`` is created.

    Args:
        L: Linear form or a list of linear forms.
        kind: PETSc vector type (``VecType``) to create.

    Returns:
        A PETSc vector with a layout that is compatible with ``L``.
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
        elif kind in (None, PETSc.Vec.Type.MPI):
            return _cpp.fem.petsc.create_vector_block(maps)
        else:
            raise NotImplementedError(f"Vector type '{kind}' not supported.")


# -- Matrix instantiation ----------------------------------------------------


def create_matrix(
    a: typing.Union[Form, Iterable[Iterable[Form]]],
    kind: typing.Optional[typing.Union[str, Iterable[Iterable[str]]]] = None,
) -> PETSc.Mat:
    """Create a PETSc matrix that is compatible with the (sequence) of bilinear form(s).

    Args:
        a: A bilinear form or a nested list of bilinear forms.
        kind: The PETSc matrix type (``MatType``). If not supplied
            and the bilinear form ``a`` is not a nested list, create a
            standard PETSc matrix. If both ``a`` and ``kind`` are a
            nested lists, create a nested PETSc matrix where each block
            ``A[i][j]`` is of type ``kind[i][j]``. If ``kind`` is
            ``PETSc.Mat.Type.NEST``, create a PETSc nest matrix. If
            ``kind`` is not supplied and ``a`` is a nested list create a
            blocked matrix.
    """
    try:
        return _cpp.fem.petsc.create_matrix(a._cpp_object, kind)  # Single form
    except AttributeError:  # ``a``` is a nested list
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
    coeffs: typing.Optional[npt.NDArray, Iterable[npt.NDArray]] = None,
) -> PETSc.Vec:
    """Assemble linear form into a new PETSc vector.

    If a single linear form is passed, the form is assembled into a
    ghosted PETSc vector. If multiple forms are passed, the forms are
    assembled into a PETSc ``VECTNEST`` vector, where each nest block is
    a ghosted PETSc vector. In this case the ``b.getNestSubVec(i)`` is
    the assembled vector for ``L[i]``.

    Constants and coefficients that appear in the forms(s) can be passed
    to avoid re-computation of constants and coefficients. The functions
    :func:`dolfinx.fem.pack_constants` and
    :func:`dolfinx.fem.pack_coefficients` can be called

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

    Returns:
        An assembled vector.
    """
    try:
        b = dolfinx.la.petsc.create_vector(
            L.function_spaces[0].dofmaps(0).index_map, L.function_spaces[0].dofmaps(0).index_map_bs
        )
        return assemble_vector(b, L, constants, coeffs)
    except AttributeError:
        b = create_vector(L, "nest")
        for b_sub in b.getNestSubVecs():
            with b_sub.localForm() as b_local:
                b_local.set(0.0)
        return assemble_vector(b, L, constants, coeffs)


@assemble_vector.register(PETSc.Vec)
def _assemble_vector_vec(
    b: PETSc.Vec,
    L: typing.Union[Form, Iterable[Form]],
    constants: typing.Optional[npt.NDArray, Iterable[npt.NDArray]] = None,
    coeffs: typing.Optional[npt.NDArray, Iterable[npt.NDArray]] = None,
) -> PETSc.Vec:
    """Assemble linear form(s) into a PETSc vector.

    The vector ``b`` must have been initialized with a size/layout that
    is consistent with the linear form. If a single form is passed, then
    ``b`` should be a ghosted PETSc vector. If multiple forms are
    passed, then ``b`  must have type ``VECNEST``.

    Constants and coefficients that appear in the forms(s) can be passed
    to avoid re-computation of constants and coefficients. The functions
    :func:`dolfinx.fem.assemble.pack_constants` and
    :func:`dolfinx.fem.assemble.pack_coefficients` can be called

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
                _assemble_vector_array(b_local.array_w, L_sub, const, coeff)
        return b
    elif isinstance(L, Iterable):
        assemble_vector_block_vec_new(b, L, constants, coeffs)
    else:
        with b.localForm() as b_local:
            _assemble_vector_array(b_local.array_w, L, constants, coeffs)
        return b


# FIXME: Revise this interface
@functools.singledispatch
def assemble_vector_block(
    L: Iterable[Form],
    a: Iterable[Iterable[Form]],
    bcs: Iterable[DirichletBC] = [],
    x0: typing.Optional[PETSc.Vec] = None,
    alpha: float = 1,
    constants_L=None,
    coeffs_L=None,
    constants_a=None,
    coeffs_a=None,
) -> PETSc.Vec:
    """Assemble linear forms into a monolithic vector.

    The vector ``b`` is assembled such that locally ``b = [b_0, b_1,
    ..., b_n, b_0g, b_1g, ..., b_ng]`` where ``b_i`` is the assembled
    vector for the 'owned' degrees-of-freedom for ``L[i]`` and ``b_ig``
    are the 'unowned' (ghost) entries for ``L[i]``.

    Note:
        The vector is not finalised, i.e. ghost values are not
        accumulated.

    Args:
        L: Linear forms to assemble into a monolithic vector.
        a: Bilinear forms to apply lifting from.
        bcs: Dirichlet boundary conditions applied to the system.
        x0: Initial guess for the solution.
        alpha: Coefficient for the lifting term.
        constants_L: Constants appearing in the linear forms.
        coeffs_L: Coefficients appearing in the linear forms.
        constants_a: Constants appearing in the bilinear forms.
        constants_a: Coefficients appearing in the bilinear forms.

    Returns:
        Assembled vector.
    """
    maps = [
        (
            form.function_spaces[0].dofmaps(0).index_map,
            form.function_spaces[0].dofmaps(0).index_map_bs,
        )
        for form in L
    ]
    b = _cpp.fem.petsc.create_vector_block(maps)
    with b.localForm() as b_local:
        b_local.set(0.0)
    return assemble_vector_block(
        b, L, a, bcs, x0, alpha, constants_L, coeffs_L, constants_a, coeffs_a
    )


def apply_lifting_block(
    maps,
    b: PETSc.Vec,
    a: typing.Union[Iterable[Form], Iterable[Iterable[Form]]],
    bcs: typing.Union[Iterable[DirichletBC], Iterable[Iterable[DirichletBC]]],
    x0: typing.Optional[Iterable[PETSc.Vec]] = None,
    alpha: float = 1,
    constants=None,
    coeffs=None,
) -> None:
    x0_local = _cpp.la.petsc.get_local_vectors(x0, maps) if x0 is not None else None
    b_local = _cpp.la.petsc.get_local_vectors(b, maps)
    bcs1 = _bcs_by_block(_extract_spaces(a, 1), bcs)
    offset0 = 0
    offset1 = functools.reduce(lambda x, y: x + y, map(lambda m: m[0].size_local * m[1], maps))
    with b.localForm() as b_l:
        for bx_, size, a_ in zip(b_local, maps, a):
            const = pack_constants(a_) if constants is None else constants
            coeff = pack_coefficients(a_) if coeffs is None else coeffs

            idxmap, bs = size
            const_ = [np.empty(0, dtype=PETSc.ScalarType) if val is None else val for val in const]
            _apply_lifting(bx_, a_, bcs1, x0_local, float(alpha), const_, coeff)

            # Add to parent vector
            b_l.array_w[offset0 : offset0 + idxmap.size_local * bs] = bx_[: idxmap.size_local * bs]
            offset0 += idxmap.size_local * bs
            b_l.array_w[offset1 : offset1 + idxmap.num_ghosts * bs] = bx_[idxmap.size_local * bs :]
            offset1 += idxmap.num_ghosts * bs


def set_bc_block(
    V,
    b: PETSc.Vec,
    bcs: Iterable[DirichletBC],
    x0: typing.Optional[PETSc.Vec] = None,
    alpha: float = 1,
) -> None:
    maps = [(_V.dofmaps(0).index_map, _V.dofmaps(0).index_map_bs) for _V in V]

    x0_sub = _cpp.la.petsc.get_local_vectors(x0, maps) if x0 is not None else [None] * len(maps)
    bcs0 = _bcs_by_block(V, bcs)
    offset = 0
    b_array = b.getArray(readonly=False)
    for submap, bcs, _x0 in zip(maps, bcs0, x0_sub):
        size = submap[0].size_local * submap[1]
        for bc in bcs:
            bc.set(b_array[offset : offset + size], _x0, alpha)
        offset += size


def assemble_vector_block_vec_new(
    b: PETSc.Vec, L: Iterable[Form], constants_L=None, coeffs_L=None
) -> PETSc.Vec:
    """Assemble linear forms into a monolithic vector ``b``.

    The vector ``b`` is assembled such that locally ``b = [b_0, b_1,
    ..., b_n, b_0g, b_1g, ..., b_ng]`` where ``b_i`` is the assembled
    vector for the 'owned' degrees-of-freedom for ``L[i]`` and ``b_ig``
    are the 'unowned' (ghost) entries for ``L[i]``.

    The vector ``b`` must have been initialized with a appropriate
    size/layout.

    Note:
        The vector is not zeroed and it is not finalised, i.e. ghost values
        are not accumulated.

    Args:
        b: Vector to assemble linear forms into.
        L: Linear forms to assemble into a monolithic vector.
        a: Bilinear forms to apply lifting from.
        bcs: Dirichlet boundary conditions applied to the system.
        x0: Initial guess for the solution.
        alpha: Coefficient for the lifting term.
        constants_L: Constants appearing in the linear forms.
        coeffs_L: Coefficients appearing in the linear forms.
        constants_a: Constants appearing in the bilinear forms.
        constants_a: Coefficients appearing in the bilinear forms.

    Returns:
        Assembled vector.
    """
    maps = [
        (
            form.function_spaces[0].dofmaps(0).index_map,
            form.function_spaces[0].dofmaps(0).index_map_bs,
        )
        for form in L
    ]

    constants_L = pack_constants(L) if constants_L is None else constants_L
    coeffs_L = pack_coefficients(L) if coeffs_L is None else coeffs_L

    offset0 = 0
    offset1 = functools.reduce(lambda x, y: x + y, map(lambda m: m[0].size_local * m[1], maps))
    with b.localForm() as b_l:
        for size, L_, const_L, coeff_L in zip(maps, L, constants_L, coeffs_L):
            # Assemble
            idxmap, bs = size
            bx_ = np.zeros((idxmap.size_local + idxmap.num_ghosts) * bs, dtype=PETSc.ScalarType)
            _assemble_vector_array(bx_, L_, const_L, coeff_L)

            # Add to parent vector
            b_l.array_w[offset0 : offset0 + idxmap.size_local * bs] += bx_[: idxmap.size_local * bs]
            b_l.array_w[offset1 : offset1 + idxmap.num_ghosts * bs] += bx_[idxmap.size_local * bs :]
            offset0 += idxmap.size_local * bs
            offset1 += idxmap.num_ghosts * bs

    return b


@assemble_vector_block.register
def _assemble_vector_block_vec(
    b: PETSc.Vec,
    L: Iterable[Form],
    a: Iterable[Iterable[Form]],
    bcs: Iterable[DirichletBC] = [],
    x0: typing.Optional[PETSc.Vec] = None,
    alpha: float = 1,
    constants_L=None,
    coeffs_L=None,
    constants_a=None,
    coeffs_a=None,
) -> PETSc.Vec:
    """Assemble linear forms into a monolithic vector ``b``.

    The vector ``b`` is assembled such that locally ``b = [b_0, b_1,
    ..., b_n, b_0g, b_1g, ..., b_ng]`` where ``b_i`` is the assembled
    vector for the 'owned' degrees-of-freedom for ``L[i]`` and ``b_ig``
    are the 'unowned' (ghost) entries for ``L[i]``.

    The vector ``b`` must have been initialized with a appropriate
    size/layout.

    Note:
        The vector is not zeroed and it is not finalised, i.e. ghost values
        are not accumulated.

    Args:
        b: Vector to assemble linear forms into.
        L: Linear forms to assemble into a monolithic vector.
        a: Bilinear forms to apply lifting from.
        bcs: Dirichlet boundary conditions applied to the system.
        x0: Initial guess for the solution.
        alpha: Coefficient for the lifting term.
        constants_L: Constants appearing in the linear forms.
        coeffs_L: Coefficients appearing in the linear forms.
        constants_a: Constants appearing in the bilinear forms.
        constants_a: Coefficients appearing in the bilinear forms.

    Returns:
        Assembled vector.
    """
    assemble_vector(b, L, constants_L, coeffs_L)

    V = [form.function_spaces[0] for form in L]
    maps = [(_V.dofmaps(0).index_map, _V.dofmaps(0).index_map_bs) for _V in V]

    apply_lifting_block(maps, b, a, bcs, x0, alpha, constants_a, coeffs_a)

    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    set_bc_block(V, b, bcs, x0, alpha)

    return b


# -- Matrix assembly ---------------------------------------------------------
@functools.singledispatch
def assemble_matrix(
    a: typing.Union[Form, Iterable[Iterable[Form]]],
    bcs: Iterable[DirichletBC] = [],
    diagonal: float = 1.0,
    constants=None,
    coeffs=None,
    kind=None,
):
    """Assemble bilinear form into a matrix.

    Note:
        The returned matrix is not 'assembled', i.e. ghost contributions
        have not been communicated.

    Args:
        a: Bilinear form to assembled into a matrix.
        bc: Dirichlet boundary conditions applied to the system.
        diagonal: Value to set on the matrix diagonal for Dirichlet
            boundary condition constrained degrees-of-freedom belonging
            to the same trial and test space.
        constants: Constants appearing the in the form.
        coeffs: Coefficients appearing the in the form.

    Returns:
        Matrix representing the bilinear form.
    """
    try:
        _a = [[None if form is None else form._cpp_object for form in arow] for arow in a]
        A = _cpp.fem.petsc.create_matrix_nest(_a, kind)
        assemble_matrix(A, a, bcs, diagonal, constants, coeffs)
        return A
    except TypeError:
        A = _cpp.fem.petsc.create_matrix(a._cpp_object, kind)
        assemble_matrix(A, a, bcs, diagonal, constants, coeffs)
        return A


@assemble_matrix.register
def assemble_matrix_mat(
    A: PETSc.Mat,
    a: typing.Union[Form, Iterable[Iterable[Form]]],
    bcs: Iterable[DirichletBC] = [],
    diagonal: float = 1.0,
    constants=None,
    coeffs=None,
) -> PETSc.Mat:
    """Assemble bilinear form into a matrix.

    The returned matrix is not finalised, i.e. ghost values are not
    accumulated.
    """
    if A.getType() == PETSc.Mat.Type.NEST:
        constants = [pack_constants(forms) for forms in a] if constants is None else constants
        coeffs = [pack_coefficients(forms) for forms in a] if coeffs is None else coeffs
        for i, (a_row, const_row, coeff_row) in enumerate(zip(a, constants, coeffs)):
            for j, (a_block, const, coeff) in enumerate(zip(a_row, const_row, coeff_row)):
                if a_block is not None:
                    Asub = A.getNestSubMatrix(i, j)
                    assemble_matrix(Asub, a_block, bcs, diagonal, const, coeff)
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
        return A
    else:
        constants = pack_constants(a) if constants is None else constants
        coeffs = pack_coefficients(a) if coeffs is None else coeffs
        _bcs = [bc._cpp_object for bc in bcs]
        _cpp.fem.petsc.assemble_matrix(A, a._cpp_object, constants, coeffs, _bcs)
        if a.function_spaces[0] is a.function_spaces[1]:
            A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
            A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
            _cpp.fem.petsc.insert_diagonal(A, a.function_spaces[0], _bcs, diagonal)
        return A


# FIXME: Revise this interface
@functools.singledispatch
def assemble_matrix_block(
    a: Iterable[Iterable[Form]],
    bcs: Iterable[DirichletBC] = [],
    diagonal: float = 1.0,
    constants=None,
    coeffs=None,
) -> PETSc.Mat:
    """Assemble bilinear forms into a blocked matrix."""
    _a = [[None if form is None else form._cpp_object for form in arow] for arow in a]
    A = _cpp.fem.petsc.create_matrix_block(_a, None)
    return _assemble_matrix_block_mat(A, a, bcs, diagonal, constants, coeffs)


@assemble_matrix_block.register
def _assemble_matrix_block_mat(
    A: PETSc.Mat,
    a: Iterable[Iterable[Form]],
    bcs: Iterable[DirichletBC] = [],
    diagonal: float = 1.0,
    constants=None,
    coeffs=None,
) -> PETSc.Mat:
    """Assemble bilinear forms into a blocked matrix."""
    constants = [pack_constants(forms) for forms in a] if constants is None else constants
    coeffs = [pack_coefficients(forms) for forms in a] if coeffs is None else coeffs

    V = _extract_function_spaces(a)
    is_rows = _cpp.la.petsc.create_index_sets(
        [(Vsub.dofmaps(0).index_map, Vsub.dofmaps(0).index_map_bs) for Vsub in V[0]]
    )
    is_cols = _cpp.la.petsc.create_index_sets(
        [(Vsub.dofmaps(0).index_map, Vsub.dofmaps(0).index_map_bs) for Vsub in V[1]]
    )

    # Assemble form
    _bcs = [bc._cpp_object for bc in bcs]
    for i, a_row in enumerate(a):
        for j, a_sub in enumerate(a_row):
            if a_sub is not None:
                Asub = A.getLocalSubMatrix(is_rows[i], is_cols[j])
                _cpp.fem.petsc.assemble_matrix(
                    Asub, a_sub._cpp_object, constants[i][j], coeffs[i][j], _bcs, True
                )
                A.restoreLocalSubMatrix(is_rows[i], is_cols[j], Asub)
            elif i == j:
                for bc in bcs:
                    row_forms = [row_form for row_form in a_row if row_form is not None]
                    assert len(row_forms) > 0
                    if row_forms[0].function_spaces[0].contains(bc.function_space):
                        raise RuntimeError(
                            f"Diagonal sub-block ({i}, {j}) cannot be 'None' "
                            " and have DirichletBC applied."
                            " Consider assembling a zero block."
                        )

    # Flush to enable switch from add to set in the matrix
    A.assemble(PETSc.Mat.AssemblyType.FLUSH)

    # Set diagonal
    for i, a_row in enumerate(a):
        for j, a_sub in enumerate(a_row):
            if a_sub is not None:
                Asub = A.getLocalSubMatrix(is_rows[i], is_cols[j])
                if a_sub.function_spaces[0] is a_sub.function_spaces[1]:
                    _cpp.fem.petsc.insert_diagonal(Asub, a_sub.function_spaces[0], _bcs, diagonal)
                A.restoreLocalSubMatrix(is_rows[i], is_cols[j], Asub)

    return A


# -- Modifiers for Dirichlet conditions ---------------------------------------


def apply_lifting(
    b: PETSc.Vec,
    a: typing.Union[Iterable[Form], Iterable[Iterable[Form]]],
    bcs: typing.Union[Iterable[DirichletBC], Iterable[Iterable[DirichletBC]]],
    x0: typing.Optional[Iterable[PETSc.Vec]] = None,
    alpha: float = 1,
    constants=None,
    coeffs=None,
) -> None:
    """Apply the function :func:`dolfinx.fem.apply_lifting` to a PETSc Vector."""
    if b.getType() == PETSc.Vec.Type.NEST:
        x0 = [] if x0 is None else x0.getNestSubVecs()
        bcs1 = _bcs_by_block(_extract_spaces(a, 1), bcs)
        constants = [pack_constants(forms) for forms in a] if constants is None else constants
        coeffs = [pack_coefficients(forms) for forms in a] if coeffs is None else coeffs
        for b_sub, a_sub, const, coeff in zip(b.getNestSubVecs(), a, constants, coeffs):
            const_ = list(
                map(lambda x: np.array([], dtype=PETSc.ScalarType) if x is None else x, const)
            )
            apply_lifting(b_sub, a_sub, bcs1, x0, alpha, const_, coeff)
        return b
    else:
        with contextlib.ExitStack() as stack:
            x0 = [] if x0 is None else x0
            x0 = [stack.enter_context(x.localForm()) for x in x0]
            x0_r = [x.array_r for x in x0]
            b_local = stack.enter_context(b.localForm())
            _apply_lifting(b_local.array_w, a, bcs, x0_r, alpha, constants, coeffs)
        return b


def set_bc(
    b: PETSc.Vec,
    bcs: typing.Union[Iterable[DirichletBC], Iterable[Iterable[DirichletBC]]],
    x0: typing.Optional[PETSc.Vec] = None,
    alpha: float = 1,
) -> None:
    """Apply the function :func:`dolfinx.fem.set_bc` to a PETSc Vector."""
    if b.getType() == PETSc.Vec.Type.NEST:
        _b = b.getNestSubVecs()
        x0 = len(_b) * [None] if x0 is None else x0.getNestSubVecs()
        for b_sub, bc, x_sub in zip(_b, bcs, x0):
            set_bc(b_sub, bc, x_sub, alpha)
    else:
        x0 = x0.array_r if x0 is not None else None
        for bc in bcs:
            bc.set(b.array_w, x0, alpha)


class LinearProblem:
    """Class for solving a linear variational problem.

    Solves of the form :math:`a(u, v) = L(v) \\,  \\forall v \\in V`
    using PETSc as a linear algebra backend.
    """

    def __init__(
        self,
        a: ufl.Form,
        L: ufl.Form,
        bcs: Iterable[DirichletBC] = [],
        u: typing.Optional[_Function] = None,
        petsc_options: typing.Optional[dict] = None,
        form_compiler_options: typing.Optional[dict] = None,
        jit_options: typing.Optional[dict] = None,
    ):
        """Initialize solver for a linear variational problem.

        Args:
            a: A bilinear UFL form, the left hand side of the
                variational problem.
            L: A linear UFL form, the right hand side of the variational
                problem.
            bcs: A list of Dirichlet boundary conditions.
            u: The solution function. It will be created if not provided.
            petsc_options: Options that are passed to the linear
                algebra backend PETSc. For available choices for the
                'petsc_options' kwarg, see the `PETSc documentation
                <https://petsc4py.readthedocs.io/en/stable/manual/ksp/>`_.
            form_compiler_options: Options used in FFCx compilation of
                this form. Run ``ffcx --help`` at the commandline to see
                all available options.
            jit_options: Options used in CFFI JIT compilation of C
                code generated by FFCx. See `python/dolfinx/jit.py` for
                all available options. Takes priority over all other
                option values.

        Example::

            problem = LinearProblem(a, L, [bc0, bc1], petsc_options={"ksp_type": "preonly",
                                                                     "pc_type": "lu",
                                                                     "pc_factor_mat_solver_type":
                                                                       "mumps"})
        """
        self._a = _create_form(
            a,
            dtype=PETSc.ScalarType,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
        )
        self._A = create_matrix(self._a)
        self._L = _create_form(
            L,
            dtype=PETSc.ScalarType,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
        )
        self._b = create_vector(self._L)

        if u is None:
            # Extract function space from TrialFunction (which is at the
            # end of the argument list as it is numbered as 1, while the
            # Test function is numbered as 0)
            self.u = _Function(a.arguments()[-1].ufl_function_space())
        else:
            self.u = u

        self._x = dolfinx.la.petsc.create_vector_wrap(self.u.x)
        self.bcs = bcs

        self._solver = PETSc.KSP().create(self.u.function_space.mesh.comm)
        self._solver.setOperators(self._A)

        # Give PETSc solver options a unique prefix
        problem_prefix = f"dolfinx_solve_{id(self)}"
        self._solver.setOptionsPrefix(problem_prefix)

        # Set PETSc options
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

    def solve(self) -> _Function:
        """Solve the problem."""

        # Assemble lhs
        self._A.zeroEntries()
        assemble_matrix_mat(self._A, self._a, bcs=self.bcs)
        self._A.assemble()

        # Assemble rhs
        with self._b.localForm() as b_loc:
            b_loc.set(0)
        assemble_vector(self._b, self._L)

        # Apply boundary conditions to the rhs
        apply_lifting(self._b, [self._a], bcs=[self.bcs])
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        for bc in self.bcs:
            bc.set(self._b.array_w)

        # Solve linear system and update ghost values in the solution
        self._solver.solve(self._b, self._x)
        self.u.x.scatter_forward()

        return self.u

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


class NonlinearProblem:
    """Nonlinear problem class for solving the non-linear problems.

    Solves problems of the form :math:`F(u, v) = 0 \\ \\forall v \\in V` using
    PETSc as the linear algebra backend.
    """

    def __init__(
        self,
        F: ufl.form.Form,
        u: _Function,
        bcs: Iterable[DirichletBC] = [],
        J: ufl.form.Form = None,
        form_compiler_options: typing.Optional[dict] = None,
        jit_options: typing.Optional[dict] = None,
    ):
        """Initialize solver for solving a non-linear problem using Newton's method`.

        Args:
            F: The PDE residual F(u, v)
            u: The unknown
            bcs: List of Dirichlet boundary conditions
            J: UFL representation of the Jacobian (Optional)
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
        apply_lifting(b, [self._a], bcs=[self.bcs], x0=[x], alpha=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, self.bcs, x, -1.0)

    def J(self, x: PETSc.Vec, A: PETSc.Mat) -> None:
        """Assemble the Jacobian matrix.

        Args:
            x: The vector containing the latest solution
        """
        A.zeroEntries()
        assemble_matrix_mat(A, self._a, self.bcs)
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


def interpolation_matrix(space0: _FunctionSpace, space1: _FunctionSpace) -> PETSc.Mat:
    """Assemble an interpolation operator matrix.

    Args:
        space0: Space to interpolate from.
        space1: Space to interpolate into.

    Returns:
        Interpolation matrix.
    """
    return _interpolation_matrix(space0._cpp_object, space1._cpp_object)


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
