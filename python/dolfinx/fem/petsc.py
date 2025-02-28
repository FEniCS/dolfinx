# Copyright (C) 2018-2025 Garth N. Wells, Nathan Sime and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Assembly functions into PETSc objects for variational forms.

Functions in this module generally apply functions in :mod:`dolfinx.fem`
to PETSc linear algebra objects and handle any PETSc-specific
preparation."""

# mypy: ignore-errors

from __future__ import annotations

import contextlib
import functools
import typing

from petsc4py import PETSc

# ruff: noqa: E402
import dolfinx

assert dolfinx.has_petsc4py

import numpy as np
import numpy.typing as npt

import dolfinx.cpp as _cpp
import ufl
from dolfinx import la
from dolfinx.cpp.fem import pack_coefficients as _pack_coefficients
from dolfinx.cpp.fem import pack_constants as _pack_constants
from dolfinx.cpp.fem.petsc import discrete_curl as _discrete_curl
from dolfinx.cpp.fem.petsc import discrete_gradient as _discrete_gradient
from dolfinx.cpp.fem.petsc import interpolation_matrix as _interpolation_matrix
from dolfinx.fem import assemble as _assemble
from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.bcs import bcs_by_block as _bcs_by_block
from dolfinx.fem.forms import Form
from dolfinx.fem.forms import extract_function_spaces as _extract_spaces
from dolfinx.fem.forms import form as _create_form
from dolfinx.fem.function import Function as _Function
from dolfinx.fem.function import FunctionSpace as _FunctionSpace
from dolfinx.la import create_petsc_vector

__all__ = [
    "LinearProblem",
    "NonlinearProblem",
    "apply_lifting",
    "apply_lifting_nest",
    "assemble_matrix",
    "assemble_matrix_block",
    "assemble_matrix_nest",
    "assemble_vector",
    "assemble_vector_block",
    "assemble_vector_nest",
    "create_matrix",
    "create_matrix_block",
    "create_matrix_nest",
    "create_vector",
    "create_vector_block",
    "create_vector_nest",
    "discrete_curl",
    "discrete_gradient",
    "interpolation_matrix",
    "set_bc",
    "set_bc_nest",
]


def _extract_function_spaces(a: list[list[Form]]):
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


def create_vector(L: Form) -> PETSc.Vec:
    """Create a PETSc vector that is compatible with a linear form.

    Note:
        Due to subtle issues in the interaction between petsc4py memory management
        and the Python garbage collector, it is recommended that the method ``PETSc.Vec.destroy()``
        is called on the returned object once the object is no longer required. Note that
        ``PETSc.Vec.destroy()`` is collective over the object's MPI communicator.

    Args:
        L: A linear form.

    Returns:
        A PETSc vector with a layout that is compatible with ``L``.
    """
    dofmap = L.function_spaces[0].dofmaps(0)
    return create_petsc_vector(dofmap.index_map, dofmap.index_map_bs)


def create_vector_block(L: list[Form]) -> PETSc.Vec:
    """Create a PETSc vector (blocked) that is compatible with a list of linear forms.

    Note:
        Due to subtle issues in the interaction between petsc4py memory management
        and the Python garbage collector, it is recommended that the method ``PETSc.Vec.destroy()``
        is called on the returned object once the object is no longer required. Note that
        ``PETSc.Vec.destroy()`` is collective over the object's MPI communicator.

    Args:
        L: List of linear forms.

    Returns:
        A PETSc vector with a layout that is compatible with ``L``.

    """
    maps = [
        (
            form.function_spaces[0].dofmaps(0).index_map,
            form.function_spaces[0].dofmaps(0).index_map_bs,
        )
        for form in L
    ]
    return _cpp.fem.petsc.create_vector_block(maps)


def create_vector_nest(L: list[Form]) -> PETSc.Vec:
    """Create a PETSc nested vector (``VecNest``) that is compatible
    with a list of linear forms.

    Args:
        L: List of linear forms.

    Returns:
        A PETSc nested vector (``VecNest``) with a layout that is
        compatible with ``L``.
    """
    maps = [
        (
            form.function_spaces[0].dofmaps(0).index_map,
            form.function_spaces[0].dofmaps(0).index_map_bs,
        )
        for form in L
    ]
    return _cpp.fem.petsc.create_vector_nest(maps)


# -- Matrix instantiation ----------------------------------------------------


def create_matrix(a: Form, mat_type=None) -> PETSc.Mat:
    """Create a PETSc matrix that is compatible with a bilinear form.

    Note:
        Due to subtle issues in the interaction between petsc4py memory
        management and the Python garbage collector, it is recommended
        that the method ``PETSc.Mat.destroy()`` is called on the
        returned object once the object is no longer required. Note that
        ``PETSc.Mat.destroy()`` is collective over the object's MPI
        communicator.

    Args:
        a: A bilinear form.
        mat_type: The PETSc matrix type (``MatType``).

    Returns:
        A PETSc matrix with a layout that is compatible with ``a``.
    """
    if mat_type is None:
        return _cpp.fem.petsc.create_matrix(a._cpp_object)
    else:
        return _cpp.fem.petsc.create_matrix(a._cpp_object, mat_type)


def create_matrix_block(a: list[list[Form]]) -> PETSc.Mat:
    """Create a PETSc matrix that is compatible with a rectangular array of bilinear forms.

    Note:
        Due to subtle issues in the interaction between petsc4py memory
        management and the Python garbage collector, it is recommended
        that the method ``PETSc.Mat.destroy()`` is called on the
        returned object once the object is no longer required. Note that
        ``PETSc.Mat.destroy()`` is collective over the object's MPI
        communicator.

    Args:
        a: Rectangular array of bilinear forms.

    Returns:
        A PETSc matrix with a blocked layout that is compatible with
        ``a``.
    """
    _a = [[None if form is None else form._cpp_object for form in arow] for arow in a]
    return _cpp.fem.petsc.create_matrix_block(_a)


def create_matrix_nest(a: list[list[Form]]) -> PETSc.Mat:
    """Create a PETSc matrix (``MatNest``) that is compatible with an array of bilinear forms.

    Note:
        Due to subtle issues in the interaction between petsc4py memory
        management and the Python garbage collector, it is recommended
        that the method ``PETSc.Mat.destroy()`` is called on the
        returned object once the object is no longer required. Note that
        ``PETSc.Mat.destroy()`` is collective over the object's MPI
        communicator.

    Args:
        a: Rectangular array of bilinear forms.

    Returns:
        A PETSc matrix (``MatNest``) that is compatible with ``a``.
    """
    _a = [[None if form is None else form._cpp_object for form in arow] for arow in a]
    return _cpp.fem.petsc.create_matrix_nest(_a)


# -- Vector assembly ---------------------------------------------------------


@functools.singledispatch
def assemble_vector(L: typing.Any, constants=None, coeffs=None) -> PETSc.Vec:
    """Assemble linear form into a new PETSc vector.

    Note:
        The returned vector is not finalised, i.e. ghost values are not
        accumulated on the owning processes.

    Note:
        Due to subtle issues in the interaction between petsc4py memory management
        and the Python garbage collector, it is recommended that the method ``PETSc.Vec.destroy()``
        is called on the returned object once the object is no longer required. Note that
        ``PETSc.Vec.destroy()`` is collective over the object's MPI communicator.

    Args:
        L: A linear form.

    Returns:
        An assembled vector.
    """
    b = create_petsc_vector(
        L.function_spaces[0].dofmaps(0).index_map, L.function_spaces[0].dofmaps(0).index_map_bs
    )
    with b.localForm() as b_local:
        _assemble._assemble_vector_array(b_local.array_w, L, constants, coeffs)
    return b


@assemble_vector.register(PETSc.Vec)
def _assemble_vector_vec(b: PETSc.Vec, L: Form, constants=None, coeffs=None) -> PETSc.Vec:
    """Assemble linear form into an existing PETSc vector.

    Note:
        The vector is not zeroed before assembly and it is not
        finalised, i.e. ghost values are not accumulated on the owning
        processes.

    Args:
        b: Vector to assemble the contribution of the linear form into.
        L: A linear form to assemble into ``b``.

    Returns:
        An assembled vector.
    """
    with b.localForm() as b_local:
        _assemble._assemble_vector_array(b_local.array_w, L, constants, coeffs)
    return b


@functools.singledispatch
def assemble_vector_nest(L: typing.Any, constants=None, coeffs=None) -> PETSc.Vec:
    """Assemble linear forms into a new nested PETSc (``VecNest``) vector.

    The returned vector is not finalised, i.e. ghost values are not
    accumulated on the owning processes.

    Note:
        Due to subtle issues in the interaction between petsc4py memory
        management and the Python garbage collector, it is recommended
        that the method ``PETSc.Vec.destroy()`` is called on the
        returned object once the object is no longer required. Note that
        ``PETSc.Vec.destroy()`` is collective over the object's MPI
        communicator.
    """
    maps = [
        (
            form.function_spaces[0].dofmaps(0).index_map,
            form.function_spaces[0].dofmaps(0).index_map_bs,
        )
        for form in L
    ]
    b = _cpp.fem.petsc.create_vector_nest(maps)
    for b_sub in b.getNestSubVecs():
        with b_sub.localForm() as b_local:
            b_local.set(0.0)
    return _assemble_vector_nest_vec(b, L, constants, coeffs)


@assemble_vector_nest.register
def _assemble_vector_nest_vec(
    b: PETSc.Vec, L: list[Form], constants=None, coeffs=None
) -> PETSc.Vec:
    """Assemble linear forms into a nested PETSc (``VecNest``) vector.

    The vector is not zeroed before assembly and it is not finalised,
    i.e. ghost values are not accumulated on the owning processes.
    """
    constants = [None] * len(L) if constants is None else constants
    coeffs = [None] * len(L) if coeffs is None else coeffs
    for b_sub, L_sub, const, coeff in zip(b.getNestSubVecs(), L, constants, coeffs):
        with b_sub.localForm() as b_local:
            _assemble._assemble_vector_array(b_local.array_w, L_sub, const, coeff)
    return b


# FIXME: Revise this interface
@functools.singledispatch
def assemble_vector_block(
    L: list[Form],
    a: list[list[Form]],
    bcs: list[DirichletBC] = [],
    x0: typing.Optional[PETSc.Vec] = None,
    alpha: float = 1,
    constants_L=None,
    coeffs_L=None,
    constants_a=None,
    coeffs_a=None,
) -> PETSc.Vec:
    """Assemble linear forms into a monolithic vector.

    The vector is not finalised, i.e. ghost values are not accumulated.

    Note:
        Due to subtle issues in the interaction between petsc4py memory
        management and the Python garbage collector, it is recommended
        that the method ``PETSc.Vec.destroy()`` is called on the
        returned object once the object is no longer required. Note that
        ``PETSc.Vec.destroy()`` is collective over the object's MPI
        communicator.
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
    return _assemble_vector_block_vec(
        b, L, a, bcs, x0, alpha, constants_L, coeffs_L, constants_a, coeffs_a
    )


@assemble_vector_block.register
def _assemble_vector_block_vec(
    b: PETSc.Vec,
    L: list[Form],
    a: list[list[Form]],
    bcs: list[DirichletBC] = [],
    x0: typing.Optional[PETSc.Vec] = None,
    alpha: float = 1,
    constants_L=None,
    coeffs_L=None,
    constants_a=None,
    coeffs_a=None,
) -> PETSc.Vec:
    """Assemble linear forms into a monolithic vector.

    The vector is not zeroed and it is not finalised, i.e. ghost values
    are not accumulated.
    """
    maps = [
        (
            form.function_spaces[0].dofmaps(0).index_map,
            form.function_spaces[0].dofmaps(0).index_map_bs,
        )
        for form in L
    ]
    if x0 is not None:
        x0_local = _cpp.la.petsc.get_local_vectors(x0, maps)
        x0_sub = x0_local
    else:
        x0_local = []
        x0_sub = [None] * len(maps)

    constants_L = (
        [form and _pack_constants(form._cpp_object) for form in L]
        if constants_L is None
        else constants_L
    )
    coeffs_L = (
        [{} if form is None else _pack_coefficients(form._cpp_object) for form in L]
        if coeffs_L is None
        else coeffs_L
    )

    constants_a = (
        [
            [
                _pack_constants(form._cpp_object)
                if form is not None
                else np.array([], dtype=PETSc.ScalarType)
                for form in forms
            ]
            for forms in a
        ]
        if constants_a is None
        else constants_a
    )

    coeffs_a = (
        [
            [{} if form is None else _pack_coefficients(form._cpp_object) for form in forms]
            for forms in a
        ]
        if coeffs_a is None
        else coeffs_a
    )

    _bcs = [bc._cpp_object for bc in bcs]
    bcs1 = _bcs_by_block(_extract_spaces(a, 1), _bcs)
    b_local = _cpp.la.petsc.get_local_vectors(b, maps)
    for b_sub, L_sub, a_sub, const_L, coeff_L, const_a, coeff_a in zip(
        b_local, L, a, constants_L, coeffs_L, constants_a, coeffs_a
    ):
        _cpp.fem.assemble_vector(b_sub, L_sub._cpp_object, const_L, coeff_L)
        _a_sub = [None if form is None else form._cpp_object for form in a_sub]
        _cpp.fem.apply_lifting(b_sub, _a_sub, const_a, coeff_a, bcs1, x0_local, alpha)

    _cpp.la.petsc.scatter_local_vectors(b, b_local, maps)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    bcs0 = _bcs_by_block(_extract_spaces(L), _bcs)
    offset = 0
    b_array = b.getArray(readonly=False)
    for submap, bcs, _x0 in zip(maps, bcs0, x0_sub):
        size = submap[0].size_local * submap[1]
        for bc in bcs:
            bc.set(b_array[offset : offset + size], _x0, alpha)
        offset += size

    return b


# -- Matrix assembly ---------------------------------------------------------
@functools.singledispatch
def assemble_matrix(
    a: typing.Any, bcs: list[DirichletBC] = [], diagonal: float = 1.0, constants=None, coeffs=None
):
    """Assemble bilinear form into a matrix.

    The returned matrix is not finalised, i.e. ghost values are not
    accumulated.

    Note:
        The returned matrix is not 'assembled', i.e. ghost contributions
        have not been communicated.

    Note:
        Due to subtle issues in the interaction between petsc4py memory
        management and the Python garbage collector, it is recommended
        that the method ``PETSc.Mat.destroy()`` is called on the
        returned object once the object is no longer required. Note that
        ``PETSc.Mat.destroy()`` is collective over the object's MPI
        communicator.

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
    A = _cpp.fem.petsc.create_matrix(a._cpp_object)
    assemble_matrix_mat(A, a, bcs, diagonal, constants, coeffs)
    return A


@assemble_matrix.register
def assemble_matrix_mat(
    A: PETSc.Mat,
    a: Form,
    bcs: list[DirichletBC] = [],
    diagonal: float = 1.0,
    constants=None,
    coeffs=None,
) -> PETSc.Mat:
    """Assemble bilinear form into a matrix.

    The returned matrix is not finalised, i.e. ghost values are not
    accumulated.
    """
    constants = _pack_constants(a._cpp_object) if constants is None else constants
    coeffs = _pack_coefficients(a._cpp_object) if coeffs is None else coeffs
    _bcs = [bc._cpp_object for bc in bcs]
    _cpp.fem.petsc.assemble_matrix(A, a._cpp_object, constants, coeffs, _bcs)
    if a.function_spaces[0] is a.function_spaces[1]:
        A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
        _cpp.fem.petsc.insert_diagonal(A, a.function_spaces[0], _bcs, diagonal)
    return A


# FIXME: Revise this interface
@functools.singledispatch
def assemble_matrix_nest(
    a: list[list[Form]],
    bcs: list[DirichletBC] = [],
    mat_types=[],
    diagonal: float = 1.0,
    constants=None,
    coeffs=None,
) -> PETSc.Mat:
    """Create a nested matrix and assemble bilinear forms into the matrix.

    Note:
        Due to subtle issues in the interaction between petsc4py memory
        management and the Python garbage collector, it is recommended
        that the method ``PETSc.Mat.destroy()`` is called on the
        returned object once the object is no longer required. Note that
        ``PETSc.Mat.destroy()`` is collective over the object's MPI
        communicator.

    Args:
        a: Rectangular (list-of-lists) array for bilinear forms.
        bcs: Dirichlet boundary conditions.
        mat_types: PETSc matrix type for each matrix block.
        diagonal: Value to set on the matrix diagonal for Dirichlet
            boundary condition constrained degrees-of-freedom belonging
            to the same trial and test space.
        constants: Constants appearing the in the form.
        coeffs: Coefficients appearing the in the form.

    Returns:
        PETSc matrix (``MatNest``) representing the block of bilinear
        forms.
    """
    _a = [[None if form is None else form._cpp_object for form in arow] for arow in a]
    A = _cpp.fem.petsc.create_matrix_nest(_a, mat_types)
    _assemble_matrix_nest_mat(A, a, bcs, diagonal, constants, coeffs)
    return A


@assemble_matrix_nest.register
def _assemble_matrix_nest_mat(
    A: PETSc.Mat,
    a: list[list[Form]],
    bcs: list[DirichletBC] = [],
    diagonal: float = 1.0,
    constants=None,
    coeffs=None,
) -> PETSc.Mat:
    """Assemble bilinear forms into a nested matrix

    Args:
        A: PETSc ``MatNest`` matrix. Matrix must have been correctly
            initialized for the bilinear forms.
        a: Rectangular (list-of-lists) array for bilinear forms.
        bcs: Dirichlet boundary conditions.
        mat_types: PETSc matrix type for each matrix block.
        diagonal: Value to set on the matrix diagonal for Dirichlet
            boundary condition constrained degrees-of-freedom belonging
            to the same trial and test space.
        constants: Constants appearing the in the form.
        coeffs: Coefficients appearing the in the form.

    Returns:
        PETSc matrix (``MatNest``) representing the block of bilinear
        forms.
    """
    constants = (
        [[form and _pack_constants(form._cpp_object) for form in forms] for forms in a]
        if constants is None
        else constants
    )
    coeffs = (
        [
            [{} if form is None else _pack_coefficients(form._cpp_object) for form in forms]
            for forms in a
        ]
        if coeffs is None
        else coeffs
    )
    for i, (a_row, const_row, coeff_row) in enumerate(zip(a, constants, coeffs)):
        for j, (a_block, const, coeff) in enumerate(zip(a_row, const_row, coeff_row)):
            if a_block is not None:
                Asub = A.getNestSubMatrix(i, j)
                assemble_matrix_mat(Asub, a_block, bcs, diagonal, const, coeff)
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


# FIXME: Revise this interface
@functools.singledispatch
def assemble_matrix_block(
    a: list[list[Form]],
    bcs: list[DirichletBC] = [],
    diagonal: float = 1.0,
    constants=None,
    coeffs=None,
) -> PETSc.Mat:  # type: ignore
    """Assemble bilinear forms into a blocked matrix.

    Note:
        Due to subtle issues in the interaction between petsc4py memory
        management and the Python garbage collector, it is recommended
        that the method ``PETSc.Mat.destroy()`` is called on the
        returned object once the object is no longer required. Note that
        ``PETSc.Mat.destroy()`` is collective over the object's MPI
        communicator.
    """
    _a = [[None if form is None else form._cpp_object for form in arow] for arow in a]
    A = _cpp.fem.petsc.create_matrix_block(_a)
    return _assemble_matrix_block_mat(A, a, bcs, diagonal, constants, coeffs)


@assemble_matrix_block.register
def _assemble_matrix_block_mat(
    A: PETSc.Mat,
    a: list[list[Form]],
    bcs: list[DirichletBC] = [],
    diagonal: float = 1.0,
    constants=None,
    coeffs=None,
) -> PETSc.Mat:
    """Assemble bilinear forms into a blocked matrix."""
    constants = (
        [
            [
                _pack_constants(form._cpp_object)
                if form is not None
                else np.array([], dtype=PETSc.ScalarType)
                for form in forms
            ]
            for forms in a
        ]
        if constants is None
        else constants
    )
    coeffs = (
        [
            [{} if form is None else _pack_coefficients(form._cpp_object) for form in forms]
            for forms in a
        ]
        if coeffs is None
        else coeffs
    )

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
    a: list[Form],
    bcs: list[list[DirichletBC]],
    x0: list[PETSc.Vec] = [],
    alpha: float = 1,
    constants=None,
    coeffs=None,
) -> None:
    """Apply the function :func:`dolfinx.fem.apply_lifting` to a PETSc Vector."""
    with contextlib.ExitStack() as stack:
        x0 = [stack.enter_context(x.localForm()) for x in x0]
        x0_r = [x.array_r for x in x0]
        b_local = stack.enter_context(b.localForm())
        _assemble.apply_lifting(b_local.array_w, a, bcs, x0_r, alpha, constants, coeffs)


def apply_lifting_nest(
    b: PETSc.Vec,
    a: list[list[Form]],
    bcs: list[DirichletBC],
    x0: typing.Optional[PETSc.Vec] = None,
    alpha: float = 1,
    constants=None,
    coeffs=None,
) -> PETSc.Vec:
    """Apply the function :func:`dolfinx.fem.apply_lifting` to each sub-vector
    in a nested PETSc Vector."""
    x0 = [] if x0 is None else x0.getNestSubVecs()
    bcs1 = _bcs_by_block(_extract_spaces(a, 1), bcs)
    constants = (
        [
            [
                _pack_constants(form._cpp_object)
                if form is not None
                else np.array([], dtype=PETSc.ScalarType)
                for form in forms
            ]
            for forms in a
        ]
        if constants is None
        else constants
    )
    coeffs = (
        [
            [{} if form is None else _pack_coefficients(form._cpp_object) for form in forms]
            for forms in a
        ]
        if coeffs is None
        else coeffs
    )
    for b_sub, a_sub, const, coeff in zip(b.getNestSubVecs(), a, constants, coeffs):
        apply_lifting(b_sub, a_sub, bcs1, x0, alpha, const, coeff)
    return b


def set_bc(
    b: PETSc.Vec, bcs: list[DirichletBC], x0: typing.Optional[PETSc.Vec] = None, alpha: float = 1
) -> None:
    """Apply the function :func:`dolfinx.fem.set_bc` to a PETSc Vector."""
    if x0 is not None:
        x0 = x0.array_r
    for bc in bcs:
        bc.set(b.array_w, x0, alpha)


def set_bc_nest(
    b: PETSc.Vec,
    bcs: list[list[DirichletBC]],
    x0: typing.Optional[PETSc.Vec] = None,
    alpha: float = 1,
) -> None:
    """Apply the function :func:`dolfinx.fem.set_bc` to each sub-vector
    of a nested PETSc Vector.
    """
    _b = b.getNestSubVecs()
    x0 = len(_b) * [None] if x0 is None else x0.getNestSubVecs()
    for b_sub, bc, x_sub in zip(_b, bcs, x0):
        set_bc(b_sub, bc, x_sub, alpha)


class LinearProblem:
    """Class for solving a linear variational problem.

    Solves of the form :math:`a(u, v) = L(v) \\,  \\forall v \\in V`
    using PETSc as a linear algebra backend.
    """

    def __init__(
        self,
        a: ufl.Form,
        L: ufl.Form,
        bcs: list[DirichletBC] = [],
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

        self._x = la.create_petsc_vector_wrap(self.u.x)
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
        """The compiled linear form `F`."""
        return self._L

    @property
    def a(self) -> Form:
        """The compiled bilinear form of `a`."""
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
        bcs: list[DirichletBC] = [],
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
        """The compiled linear form `F`."""
        return self._L

    @property
    def a(self) -> Form:
        """The compiled bilinear form of the Jacobian `J`"""
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

    Note:
        Due to subtle issues in the interaction between petsc4py memory
        management and the Python garbage collector, it is recommended
        that the method ``PETSc.Mat.destroy()`` is called on the
        returned object once the object is no longer required. Note that
        ``PETSc.Mat.destroy()`` is collective over the object's MPI
        communicator.

    Args:
        space0: H1 space to interpolate the gradient from.
        space1: H(curl) space to interpolate into.

    Returns:
        Discrete gradient operator.
    """
    return _discrete_gradient(space0._cpp_object, space1._cpp_object)


def interpolation_matrix(space0: _FunctionSpace, space1: _FunctionSpace) -> PETSc.Mat:
    """Assemble an interpolation operator matrix.

    Note:
        Due to subtle issues in the interaction between petsc4py memory
        management and the Python garbage collector, it is recommended
        that the method ``PETSc.Mat.destroy()`` is called on the
        returned object once the object is no longer required. Note that
        ``PETSc.Mat.destroy()`` is collective over the object's MPI
        communicator.

    Args:
        space0: Space to interpolate from.
        space1: Space to interpolate into.

    Returns:
        Interpolation matrix.
    """
    return _interpolation_matrix(space0._cpp_object, space1._cpp_object)


def assign(
    x0: typing.Union[npt.NDArray[np.floating], list[npt.NDArray[np.floating]]], x1: PETSc.Vec
):
    """Assign x0 to x1.

    Assigns values in ``x0``, which is possibly blocked, to ``x1``. When
    ``x0`` holds a list of arrays, the arrays in ``x0`` are 'stacked'
    and assigned to ``x1``, i.e.::

              [x0[0]]
        x1 =  [x0[1]]
              [.....]
              [x0[n-1]]

    Args:
        x0: An array or list of array that will be assigned to ``x1``.
        x1: Vector to assign to.
    """
    try:
        # Nested PETSc matrix
        x1_nest = x1.getNestSubVecs()
        for _x0, _x1 in zip(x0, x1_nest):
            with _x1.localForm() as x:
                x.array_w[:] = _x0
    except AttributeError:
        with x1.localForm() as _x:
            try:
                start = 0
                for _x0 in x0:
                    end = start + _x0.shape[0]
                    _x.array_w[start:end] = _x0
                    start = end

                # The above doesn't update ghost values, so we need to
                # do it here. We should handle ghost by packing them at
                # the end of x0.
                x1.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            except:  # noqa: E722
                # TODO: add correct exception
                _x.array_w[:] = _x0


def copy_vec_to_function(
    x: PETSc.Vec, u: typing.Union[dolfinx.fem.Function, list[dolfinx.fem.Function]]
):  # type: ignore
    """Copy values of x into vectors backing function(s) u.

    MPI collective operation. Modifies u in place.

    Args:
        x: A vector, can be nested, blocked or normal.
        u: A function or list of functions.
    """
    if x.getType() == PETSc.Vec.Type().NEST:
        _copy_nest_vec_to_functions(x, u)
    elif isinstance(u, list):
        # DOLFINx-created block Vec cannot be discerned from standard SEQ and
        # MPI types
        _copy_block_vec_to_functions(x, u)
    else:
        # Fall through to standard copy
        _copy_vec_to_function(x, u)


def _copy_vec_to_function(x: PETSc.Vec, u: dolfinx.fem.Function):  # type: ignore
    """Copy values of x into vectors backing function(s) u. Normal version."""
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
    x.copy(u.x.petsc_vec)
    u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore


def _copy_block_vec_to_functions(x: PETSc.Vec, u: list[dolfinx.fem.Function]):  # type: ignore
    """Copy values of x into vectors backing function(s) u. Block version."""
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
    offset_start = 0
    for ui in u:
        Vi = ui.function_space
        num_sub_dofs = Vi.dofmap.index_map.size_local * Vi.dofmap.index_map_bs
        ui.x.petsc_vec.array_w[:num_sub_dofs] = x.array_r[
            offset_start : offset_start + num_sub_dofs
        ]
        ui.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
        offset_start += num_sub_dofs


def _copy_nest_vec_to_functions(x: PETSc.Vec, u: list[dolfinx.fem.Function]):  # type: ignore
    """Copy values of x into vectors backing function(s) u. Nest version."""
    subvecs = x.getNestSubVecs()
    for x_sub, var_sub in zip(subvecs, u):
        x_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
        with x_sub.localForm() as _x:
            var_sub.x.array[:] = _x.array_r


def copy_function_to_vec(
    u: typing.Union[dolfinx.fem.Function, list[dolfinx.fem.Function]], x: PETSc.Vec
):  # type: ignore
    """Copy vectors backing function(s) u into vector x.

    MPI collective operation. Modifies x in place.

    Args:
        u: A function or list of functions.
        x: A vector, can be nested, blocked or normal.
    """
    if x.getType() == PETSc.Vec.Type().NEST:
        assert isinstance(u, list)
        _copy_functions_to_nest_vec(u, x)
    elif isinstance(u, list):
        # DOLFINx-created block Vec cannot be discerned from standard SEQ and
        # MPI types
        _copy_functions_to_block_vec(u, x)
    else:
        # Fall through to standard copy
        _copy_function_to_vec(u, x)


def _copy_function_to_vec(u: dolfinx.fem.Function, x: PETSc.Vec):  # type: ignore
    """Copy vectors backing function u into vector x. Normal version."""
    u.x.petsc_vec.copy(x)


def _copy_functions_to_block_vec(u: list[dolfinx.fem.Function], x: PETSc.Vec):  # type: ignore
    """Copy vectors backing function(s) u into vector x. Block version."""
    u_petsc = [ui.x.petsc_vec.array_r for ui in u]
    index_maps = [
        (ui.function_space.dofmap.index_map, ui.function_space.dofmap.index_map_bs) for ui in u
    ]
    _cpp.la.petsc.scatter_local_vectors(
        x,
        u_petsc,
        index_maps,
    )
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore


def _copy_functions_to_nest_vec(u: list[dolfinx.fem.Function], x: PETSc.Vec):  # type: ignore
    """Copy vectors backing function u into vector x. Nest version."""
    wrapped_sol = [dolfinx.la.create_petsc_vector_wrap(u_i.x) for u_i in u]
    u_nest = PETSc.Vec().createNest(wrapped_sol)  # type: ignore
    u_nest.copy(x)
    u_nest.destroy()
    [wrapped.destroy() for wrapped in wrapped_sol]
