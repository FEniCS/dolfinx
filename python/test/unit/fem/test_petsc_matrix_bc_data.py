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


@pytest.mark.parametrize("layout", ["single", "blocked", "nest", "is", "nest_is"])
def test_nonlinear_problem_reuses_bc_data(layout, monkeypatch):
    """Newton callbacks reuse prepared BC data and match fresh assembly.

    Coefficients, boundary values and the conditions themselves change
    between evaluations; each callback result is compared with the
    public assembly functions, which prepare everything afresh.
    """
    msh = mesh.create_unit_square(
        MPI.COMM_WORLD, 4, 4, dtype=PETSc.RealType, ghost_mode=mesh.GhostMode.none
    )
    V = fem.functionspace(msh, ("Lagrange", 1))
    block = layout in ("blocked", "nest", "nest_is")
    spaces = [V, V.clone()] if block else [V]
    functions = [fem.Function(W) for W in spaces]
    coefficient = fem.Constant(msh, PETSc.ScalarType(2))
    residuals = [
        ufl.inner(coefficient * u + u**3, ufl.TestFunction(W)) * ufl.dx
        for u, W in zip(functions, spaces, strict=True)
    ]
    preconditioners = [
        coefficient * ufl.inner(ufl.TrialFunction(W), ufl.TestFunction(W)) * ufl.dx for W in spaces
    ]
    P = [[preconditioners[0], None], [None, preconditioners[1]]] if block else preconditioners[0]
    kind = {
        "single": None,
        "blocked": "mpi",
        "nest": "nest",
        "is": "is",
        "nest_is": [["is", None], [None, "is"]],
    }[layout]
    boundary = fem.Constant(msh, PETSc.ScalarType(3))
    left = fem.locate_dofs_geometrical(V, lambda x: x[0] < 1.0e-3)
    right = fem.locate_dofs_geometrical(V, lambda x: x[0] > 1 - 1.0e-3)
    conditions = [
        fem.dirichletbc(boundary, dofs, W)
        for dofs, W in zip((left, right)[: len(spaces)], spaces, strict=True)
    ]
    problem = petsc.NonlinearProblem(
        residuals if block else residuals[0],
        functions if block else functions[0],
        bcs=conditions,
        P=P,
        kind=kind,
        petsc_options_prefix=f"test_nonlinear_reuse_{layout}_",
    )
    assert problem.preconditioner is not None and problem.P_mat is not None
    A_ref = petsc.create_matrix(problem.J, kind=kind)
    P_ref = petsc.create_matrix(problem.preconditioner, kind=kind)
    b_ref = problem.b.duplicate()
    eps = 1000 * np.finfo(PETSc.RealType).eps

    def unexpected(*args, **kwargs):
        raise AssertionError("Newton callbacks must not rebuild BC data")

    def compare(actual, expected):
        if actual.getType() == PETSc.Mat.Type.NEST:
            for i in range(len(spaces)):
                for j in range(len(spaces)):
                    if actual.getNestSubMatrix(i, j):
                        compare(actual.getNestSubMatrix(i, j), expected.getNestSubMatrix(i, j))
            return
        a = actual.copy().convert(PETSc.Mat.Type.AIJ)
        e = expected.copy().convert(PETSc.Mat.Type.AIJ)
        try:
            a.axpy(-1, e, PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            assert a.norm() <= eps * max(1, e.norm())
        finally:
            a.destroy()
            e.destroy()

    try:
        # Reassignment also removes formerly constrained dofs
        for bcs in (conditions, conditions[1:], []):
            problem.bcs = bcs
            for step in (1, 2):
                boundary.value = PETSc.ScalarType(step + 3)
                coefficient.value = PETSc.ScalarType(step + 1)
                for i, u in enumerate(functions):
                    u.x.array[:] = PETSc.ScalarType(step + i)
                petsc.assign(problem.u, problem.x)
                with monkeypatch.context() as patch:
                    for name in (
                        "_matrix_bc_data",
                        "_matrix_diag_data",
                        "_matis_diag_data",
                        "_block_index_sets",
                        "_lifting_bc_markers",
                        "_bcs_by_block",
                    ):
                        patch.setattr(petsc, name, unexpected)
                    problem.solver.computeFunction(problem.x, problem.b)
                    problem.solver.computeJacobian(problem.x, problem.A, problem.P_mat)

                petsc.assemble_residual(
                    problem.solver,
                    problem.x,
                    b_ref,
                    problem.u,
                    problem.F,
                    problem.J,
                    problem.bcs,
                    problem.b.getAttr("_blocks"),
                )
                petsc.assemble_jacobian(
                    problem.solver,
                    problem.x,
                    A_ref,
                    P_ref,
                    problem.u,
                    problem.J,
                    problem.preconditioner,
                    problem.bcs,
                )
                compare(problem.A, A_ref)
                compare(problem.P_mat, P_ref)
                b_ref.axpy(-1, problem.b)
                assert b_ref.norm() <= eps * max(1, problem.b.norm())
    finally:
        b_ref.destroy()
        A_ref.destroy()
        P_ref.destroy()
