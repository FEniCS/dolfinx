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


def _assert_equal(A, B):
    """Assert two assembled PETSc matrices are equal."""
    result, expected = A.copy().convert("aij"), B.copy().convert("aij")
    try:
        result.axpy(-1, expected)
        assert result.norm() < 1e-12
    finally:
        result.destroy()
        expected.destroy()


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
    for field in dataclasses.fields(data):
        value = getattr(data, field.name)
        assert value is None or isinstance(value, tuple)
    if data.index_sets is not None:
        assert all(isinstance(sets, tuple) for sets in data.index_sets)
    for array in (
        *data.row_markers,
        *data.column_markers,
        *data.diagonal_rows,
        *data.diagonal_values,
    ):
        if isinstance(array, np.ndarray):
            assert not array.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                array[:] = 0

    def unexpected(*args, **kwargs):
        raise AssertionError("Prepared assembly must not rebuild BC data")

    monkeypatch.setattr(petsc, "_matrix_bc_data", unexpected)
    monkeypatch.setattr(petsc, "_matrix_diag_data", unexpected)
    monkeypatch.setattr(petsc, "_block_index_sets", unexpected)

    matrices = [reference, A]
    try:
        for _ in range(2):
            A.zeroEntries()
            petsc.assemble_matrix(A, a, bcs=data)
            A.assemble()
            _assert_equal(A, reference)
        B = petsc.assemble_matrix(a, bcs=data, kind=matrix_kind)
        matrices.append(B)
        B.assemble()
        _assert_equal(B, reference)
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
            data, column_markers=tuple(np.zeros(1, dtype=np.int8) for _ in spaces)
        )
        with pytest.raises(ValueError, match="BC markers"):
            petsc.assemble_matrix(A, a, bcs=invalid)
        num_rows = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
        invalid = dataclasses.replace(
            data, diagonal_rows=tuple(np.array([num_rows], dtype=np.int32) for _ in spaces)
        )
        with pytest.raises(ValueError, match="Diagonal rows"):
            petsc.assemble_matrix(A, a, bcs=invalid)
        if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
            with pytest.raises(ValueError, match="Complex diagonal"):
                petsc.prepare_matrix_bcs(A, a, bcs, diag=1 + 1j)
        if kind == "aij":
            blocked = petsc.create_matrix([[a]])
            matrices.append(blocked)
            with pytest.raises(ValueError, match="single form"):
                petsc.assemble_matrix(blocked, [[a]], bcs=data)
        if kind == "is":
            other = petsc.create_matrix(a, kind="aij")
            matrices.append(other)
            with pytest.raises(ValueError, match="incompatible diagonal layout"):
                petsc.assemble_matrix(other, a, bcs=data)
    finally:
        for matrix in matrices:
            matrix.destroy()


@pytest.mark.parametrize(
    "source_kind, target_kind",
    [
        ("aij", "aij"),
        ("is", "is"),
        ("nest", "nest"),
        ("nest", "blocked"),
        ("blocked", "nest"),
        ("blocked", "blocked"),
    ],
)
@pytest.mark.parametrize("shape", [(), (2,)])
def test_prepared_matrix_bcs_different_form(source_kind, target_kind, shape, monkeypatch):
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3, ghost_mode=mesh.GhostMode.none)
    V = fem.functionspace(msh, ("Lagrange", 1, shape))
    spaces = [V, V.clone()] if source_kind in ("nest", "blocked") else [V]
    mass = []
    stiffness = []
    bcs = []
    for T in spaces:
        mass_row, stiffness_row = [], []
        for U in spaces:
            u, v = ufl.TrialFunction(U), ufl.TestFunction(T)
            mass_row.append(fem.form(ufl.inner(u, v) * ufl.dx))
            stiffness_row.append(fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx))
        mass.append(mass_row)
        stiffness.append(stiffness_row)
        dofs = fem.locate_dofs_geometrical(T, lambda x: np.isclose(x[0], 0))
        bcs.append(fem.dirichletbc(np.zeros(shape, dtype=PETSc.ScalarType), dofs, T))
    if len(spaces) == 1:
        mass, stiffness = mass[0][0], stiffness[0][0]
    source = petsc.create_matrix(mass, kind="aij" if source_kind == "blocked" else source_kind)
    target = petsc.create_matrix(stiffness, kind="aij" if target_kind == "blocked" else target_kind)
    reference = petsc.assemble_matrix(
        stiffness, bcs=bcs, kind="aij" if target_kind == "blocked" else target_kind
    )
    reference.assemble()
    data = petsc.prepare_matrix_bcs(source, mass, bcs)

    def unexpected(*args, **kwargs):
        raise AssertionError("Reusing prepared data must not rebuild BC data")

    monkeypatch.setattr(petsc, "_matrix_bc_data", unexpected)
    monkeypatch.setattr(petsc, "_matrix_diag_data", unexpected)
    monkeypatch.setattr(petsc, "_block_index_sets", unexpected)
    try:
        for _ in range(2):
            target.zeroEntries()
            petsc.assemble_matrix(target, stiffness, bcs=data)
            target.assemble()
            _assert_equal(target, reference)
    finally:
        source.destroy()
        target.destroy()
        reference.destroy()


@pytest.mark.parametrize("kind", [None, "mpi", "nest"])
def test_nonlinear_problem_reuses_bc_data(kind, monkeypatch):
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = fem.functionspace(msh, ("Lagrange", 1))
    spaces = [V] if kind is None else [V, V.clone()]
    bcs = []
    for W in spaces:
        dofs = fem.locate_dofs_geometrical(W, lambda x: np.isclose(x[0], 0))
        bcs.append(fem.dirichletbc(PETSc.ScalarType(1), dofs, W))
    eps = 1000 * np.finfo(PETSc.RealType).eps

    def make_problem(label):
        u = [fem.Function(W) for W in spaces]
        F = [
            ufl.inner(ui**2 - 2, ufl.TestFunction(W)) * ufl.dx
            for ui, W in zip(u, spaces, strict=True)
        ]
        return petsc.NonlinearProblem(
            F[0] if kind is None else F,
            u[0] if kind is None else u,
            bcs=bcs,
            kind=kind,
            petsc_options_prefix=f"test_nonlinear_reuse_{kind}_{label}_",
            petsc_options={
                "snes_rtol": eps,
                "snes_error_if_not_converged": True,
                "ksp_type": "gmres",
                "ksp_rtol": eps,
                "pc_type": "none",
            },
        )

    reference = make_problem("reference")
    problem = make_problem("reuse")
    for p in (reference, problem):
        for ui in [p.u] if kind is None else p.u:
            ui.x.array[:] = 1
    reference.solve()

    def unexpected(*args, **kwargs):
        raise AssertionError("Newton steps must not rebuild or re-check BC data")

    for name in (
        "_bcs_by_block",
        "_check_preconditioner_spaces",
        "_check_prepared_matrix_bcs",
        "_matrix_bc_data",
        "_matrix_diag_data",
        "_block_index_sets",
        "_lifting_bc_markers",
    ):
        monkeypatch.setattr(petsc, name, unexpected)
    problem.solve()
    actual = [problem.u] if kind is None else problem.u
    expected = [reference.u] if kind is None else reference.u
    for u, u_ref in zip(actual, expected, strict=True):
        np.testing.assert_allclose(u.x.array, u_ref.x.array, rtol=1e-10, atol=1e-10)
