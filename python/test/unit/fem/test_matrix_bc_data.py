# Copyright (C) 2026 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tests for reusable CSR matrix boundary-condition data."""

import dataclasses

from mpi4py import MPI

import numpy as np
import pytest

import ufl
from dolfinx import fem, mesh


@pytest.mark.parametrize("shape", [(), (2,)])
@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        pytest.param(np.complex64, marks=pytest.mark.xfail_win32_complex),
        pytest.param(np.complex128, marks=pytest.mark.xfail_win32_complex),
    ],
)
def test_prepared_matrix_bcs(shape, constrained, dtype, monkeypatch):
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4, dtype=dtype(0).real.dtype)
    V = fem.functionspace(msh, ("Lagrange", 1, shape))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = fem.form(ufl.inner(u, v) * ufl.dx, dtype=dtype)
    dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
    value = np.zeros(shape, dtype=dtype)
    bc = fem.dirichletbc(value, dofs, V)
    bcs = [bc, bc] if constrained else []
    diagonal = dtype(2 + 1j) if np.issubdtype(dtype, np.complexfloating) else dtype(2)

    default = fem.assemble_matrix(a, bcs=bcs)
    default_data = fem.prepare_matrix_bcs(a, bcs)
    default_prepared = fem.assemble_matrix(a, bcs=default_data)
    np.testing.assert_allclose(default.data, default_prepared.data)
    with pytest.warns(DeprecationWarning, match="diag argument") as record:
        reference = fem.assemble_matrix(a, bcs=bcs, diag=diagonal)
    assert record[0].filename == __file__
    reference.scatter_reverse()
    with pytest.warns(DeprecationWarning, match="diag argument") as record:
        fem.assemble_matrix(fem.create_matrix(a), a, bcs=bcs, diag=diagonal)
    assert record[0].filename == __file__
    data = fem.prepare_matrix_bcs(a, bcs, diag=diagonal)
    data = dataclasses.replace(
        data, diagonal_values=np.full(data.diagonal_rows.size, diagonal, dtype=dtype)
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("Prepared assembly must not rebuild BC data")

    monkeypatch.setattr("dolfinx.fem.assemble._bc_dof_markers_by_space", unexpected)
    monkeypatch.setattr("dolfinx.fem.assemble._owned_marked_rows", unexpected)
    A = fem.assemble_matrix(a, bcs=data)
    A.scatter_reverse()
    np.testing.assert_allclose(A.data, reference.data)
    A.data[:] = 0
    fem.assemble_matrix(A, a, bcs=data)
    A.scatter_reverse()
    np.testing.assert_allclose(A.data, reference.data)
    with pytest.raises(ValueError, match="diag cannot"):
        fem.assemble_matrix(A, a, bcs=data, diag=1)
    invalid = dataclasses.replace(data, column_marker=np.zeros(1, dtype=np.int8))
    with pytest.raises(ValueError, match="BC markers"):
        fem.assemble_matrix(A, a, bcs=invalid)
    W = V.clone()
    other = fem.form(ufl.inner(ufl.TrialFunction(W), ufl.TestFunction(W)) * ufl.dx, dtype=dtype)
    with pytest.raises(ValueError, match="different function spaces"):
        fem.assemble_matrix(A, other, bcs=data)


def test_prepared_matrix_bcs_invalid_diagonal():
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    V = fem.functionspace(msh, ("Lagrange", 1))
    a = fem.form(ufl.inner(ufl.TrialFunction(V), ufl.TestFunction(V)) * ufl.dx, dtype=np.float64)
    dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
    data = fem.prepare_matrix_bcs(a, [fem.dirichletbc(np.float64(0), dofs, V)])
    A = fem.create_matrix(a)
    num_owned = V.dofmap.index_map.size_local
    for rows, values, match in [
        (np.array([num_owned], dtype=np.int32), 1.0, "Diagonal rows"),
        (np.array([-1], dtype=np.int32), 1.0, "Diagonal rows"),
        (np.array([0], dtype=np.int64), 1.0, "Diagonal rows"),
        (np.array([0], dtype=np.int32), np.ones(2), "Diagonal values"),
        (np.array([0], dtype=np.int32), 1.0 + 1j, "Complex diagonal"),
    ]:
        invalid = dataclasses.replace(data, diagonal_rows=rows, diagonal_values=values)
        with pytest.raises(ValueError, match=match):
            fem.assemble_matrix(A, a, bcs=invalid)
    with pytest.raises(ValueError, match="Complex diagonal"):
        fem.prepare_matrix_bcs(a, [], diag=1 + 1j)


def test_prepared_matrix_bcs_rectangular():
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    V = fem.functionspace(msh, ("Lagrange", 1))
    W = fem.functionspace(msh, ("Lagrange", 2))
    a = fem.form(ufl.inner(ufl.TrialFunction(W), ufl.TestFunction(V)) * ufl.dx)
    dofs = fem.locate_dofs_geometrical(W, lambda x: np.isclose(x[0], 0))
    bc = fem.dirichletbc(np.array(0, dtype=a.dtype), dofs, W)
    data = fem.prepare_matrix_bcs(a, [bc])
    assert data.row_marker.size == 0
    assert data.column_marker.size > 0
    assert data.diagonal_rows.size == 0
    reference = fem.assemble_matrix(a, bcs=[bc])
    A = fem.assemble_matrix(a, bcs=data)
    np.testing.assert_allclose(A.data, reference.data)


def test_prepared_matrix_bcs_callback(monkeypatch):
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    V = fem.functionspace(msh, ("Lagrange", 1))
    a = fem.form(ufl.inner(ufl.TrialFunction(V), ufl.TestFunction(V)) * ufl.dx)
    dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
    bcs = [fem.dirichletbc(np.array(0, dtype=a.dtype), dofs, V)]
    data = fem.prepare_matrix_bcs(a, bcs)
    contributions = []

    def insert(rows, columns, values):
        contributions.append((rows.copy(), columns.copy(), values.copy()))
        return 0

    fem.assemble_matrix_fn(insert, a, bcs=bcs)
    reference = contributions.copy()
    contributions.clear()

    def unexpected(*args, **kwargs):
        raise AssertionError("Prepared assembly must not rebuild BC markers")

    monkeypatch.setattr("dolfinx.fem.assemble._bc_dof_markers_by_space", unexpected)
    fem.assemble_matrix_fn(insert, a, bcs=data)
    assert len(reference) == len(contributions)
    for expected, actual in zip(reference, contributions, strict=True):
        for e, r in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(e, r)
