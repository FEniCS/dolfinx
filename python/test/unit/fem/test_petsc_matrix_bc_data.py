# Copyright (C) 2026 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tests for reusable PETSc matrix boundary-condition data."""

import dataclasses

from mpi4py import MPI

import numpy as np
import pytest

import ufl
from dolfinx import fem, mesh

PETSc = pytest.importorskip("petsc4py.PETSc")
from dolfinx.fem import petsc  # noqa: E402

pytestmark = pytest.mark.petsc4py


@pytest.mark.parametrize("kind", ["aij", "is", "nest", "blocked"])
@pytest.mark.parametrize("shape", [(), (2,)])
@pytest.mark.parametrize("constrained", [False, True])
def test_prepared_matrix_bcs(kind, shape, constrained, monkeypatch):
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4, ghost_mode=mesh.GhostMode.none)
    V = fem.functionspace(msh, ("Lagrange", 1, shape))
    W = V.clone()
    spaces = [V, W] if kind in ("nest", "blocked") else [V]
    forms = [
        [fem.form(ufl.inner(ufl.TrialFunction(U), ufl.TestFunction(T)) * ufl.dx) for U in spaces]
        for T in spaces
    ]
    a = forms if len(spaces) == 2 else forms[0][0]
    bcs = []
    for U in spaces:
        dofs = fem.locate_dofs_geometrical(U, lambda x: np.isclose(x[0], 0))
        if constrained:
            bc = fem.dirichletbc(np.zeros(shape, dtype=PETSc.ScalarType), dofs, U)
            bcs.extend([bc, bc])
    matrix_kind = "aij" if kind == "blocked" else kind
    with pytest.warns(DeprecationWarning, match="diag argument") as record:
        reference = petsc.assemble_matrix(a, bcs=bcs, diag=2, kind=matrix_kind)
    assert record[0].filename == __file__
    reference.assemble()
    A = petsc.create_matrix(a, kind=matrix_kind)
    data = petsc.prepare_matrix_bcs(A, a, bcs, diag=2)

    def unexpected(*args, **kwargs):
        raise AssertionError("Prepared assembly must not rebuild BC data")

    monkeypatch.setattr(petsc, "_matrix_bc_data", unexpected)
    monkeypatch.setattr(petsc, "_matrix_diag_data", unexpected)
    monkeypatch.setattr(petsc, "_block_index_sets", unexpected)

    def assert_matches_reference(M):
        M.assemble()
        result, expected = M.copy().convert("aij"), reference.copy().convert("aij")
        try:
            result.axpy(-1, expected)
            assert result.norm() < 1e-12
        finally:
            result.destroy()
            expected.destroy()

    matrices = [reference, A]
    try:
        petsc.assemble_matrix(A, a, bcs=data)
        assert_matches_reference(A)
        A.zeroEntries()
        petsc.assemble_matrix(A, a, bcs=data)
        assert_matches_reference(A)
        B = petsc.assemble_matrix(a, bcs=data, kind=matrix_kind)
        matrices.append(B)
        assert_matches_reference(B)
        with pytest.raises(ValueError, match="diag cannot"):
            petsc.assemble_matrix(A, a, bcs=data, diag=1)
        if kind in ("aij", "is"):
            with pytest.raises(TypeError, match="different matrix type"):
                petsc.assemble_matrix(A, a, bcs=fem.prepare_matrix_bcs(a, bcs))
            with pytest.raises(TypeError, match="different matrix type"):
                fem.assemble_matrix(a, bcs=data)
            with pytest.raises(TypeError, match="different matrix type"):
                fem.assemble_matrix_fn(lambda rows, cols, vals: 0, a, bcs=data)
        invalid = dataclasses.replace(
            data,
            markers=data.markers._replace(
                column_markers=[np.zeros(1, dtype=np.int8) for _ in spaces]
            ),
        )
        with pytest.raises(ValueError, match="BC markers"):
            petsc.assemble_matrix(A, a, bcs=invalid)
        num_rows = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
        invalid = dataclasses.replace(
            data,
            diagonal=data.diagonal._replace(
                rows=[np.array([num_rows], dtype=np.int32) for _ in spaces]
            ),
        )
        with pytest.raises(ValueError, match="Diagonal rows"):
            petsc.assemble_matrix(A, a, bcs=invalid)
        if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
            with pytest.raises(ValueError, match="Complex diagonal"):
                petsc.prepare_matrix_bcs(A, a, bcs, diag=1 + 1j)
        if kind == "is":
            other = petsc.create_matrix(a, kind="aij")
            matrices.append(other)
            with pytest.raises(ValueError, match="incompatible diagonal layout"):
                petsc.assemble_matrix(other, a, bcs=data)
    finally:
        for matrix in matrices:
            matrix.destroy()
