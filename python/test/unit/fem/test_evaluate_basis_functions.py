# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tests for evaluating basis functions at points."""

from mpi4py import MPI

import numpy as np
import pytest

import basix.ufl
from dolfinx import fem, mesh


def _create_mesh(cell_type: mesh.CellType, dtype: np.dtype) -> mesh.Mesh:
    if mesh.cell_dim(cell_type) == 2:
        return mesh.create_unit_square(MPI.COMM_WORLD, 3, 4, cell_type=cell_type, dtype=dtype)
    return mesh.create_unit_cube(MPI.COMM_WORLD, 2, 3, 2, cell_type=cell_type, dtype=dtype)


def _interior_points(msh: mesh.Mesh, rng: np.random.Generator) -> np.ndarray:
    """A random convex combination of the vertices of each local cell."""
    num_cells = msh.topology.index_map(msh.topology.dim).size_local
    vertices = msh.geometry.x[msh.geometry.dofmaps[0][:num_cells]]
    weights = rng.dirichlet(np.ones(vertices.shape[1]), num_cells)
    return np.einsum("cv,cvj->cj", weights, vertices).astype(msh.geometry.x.dtype)


def _pull_back_tol(dtype: np.dtype) -> float:
    """Newton tolerance for non-affine pull-backs.

    The Newton method stops when the update is below ``tol``. It
    converges quadratically, so the remaining error is of the order of
    ``tol**2``, i.e. the machine precision of ``dtype``.
    """
    return float(np.sqrt(np.finfo(dtype).eps))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "cell_type",
    [
        mesh.CellType.triangle,
        mesh.CellType.quadrilateral,
        mesh.CellType.tetrahedron,
        mesh.CellType.hexahedron,
    ],
)
def test_kronecker_delta_at_dof_coordinates(cell_type, dtype):
    msh = _create_mesh(cell_type, dtype)
    V = fem.functionspace(msh, ("Lagrange", 2))
    dof_x = V.tabulate_dof_coordinates()

    # Evaluate at the coordinates of the dofs of each cell, in the cell's
    # local dof order
    num_cells = msh.topology.index_map(msh.topology.dim).size_local
    cell_dofs = V.dofmap.list[:num_cells]
    x = dof_x[cell_dofs.reshape(-1)]
    cells = np.repeat(np.arange(num_cells, dtype=np.int32), cell_dofs.shape[1])

    basis = fem.evaluate_basis_functions(V, x, cells, tol=_pull_back_tol(dtype))
    assert basis.shape == (x.shape[0], cell_dofs.shape[1], 1)
    assert basis.dtype == dtype

    eps = np.finfo(dtype).eps
    identity = np.tile(np.eye(cell_dofs.shape[1], dtype=dtype), (num_cells, 1))
    np.testing.assert_allclose(basis[:, :, 0], identity, atol=1e3 * eps)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("cell_type", [mesh.CellType.triangle, mesh.CellType.quadrilateral])
def test_partition_of_unity(cell_type, dtype):
    msh = _create_mesh(cell_type, dtype)
    V = fem.functionspace(msh, ("Lagrange", 3))
    rng = np.random.default_rng(13)
    x = _interior_points(msh, rng)
    cells = np.arange(x.shape[0], dtype=np.int32)

    # Skip the first point
    cells[0] = -1
    basis = fem.evaluate_basis_functions(V, x, cells, tol=_pull_back_tol(dtype))
    assert basis.dtype == dtype

    eps = np.finfo(dtype).eps
    np.testing.assert_allclose(basis[1:].sum(axis=(1, 2)), 1, atol=1e3 * eps)
    assert np.all(basis[0] == 0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize(
    "cell_type, element",
    [
        (mesh.CellType.triangle, ("Lagrange", 2, (2,))),
        (mesh.CellType.quadrilateral, ("Lagrange", 2, (2,))),
        (mesh.CellType.tetrahedron, ("N1curl", 2)),
        (mesh.CellType.hexahedron, ("N1curl", 1)),
    ],
)
def test_expansion_matches_function_eval(cell_type, element, dtype):
    """Contracting the basis with coefficients reproduces Function.eval.

    Covers the output layout for a blocked space and for an element
    needing dof transformations and a Piola map. The basis is real, in
    the geometry type, while the coefficients have the scalar type.
    """
    xtype = np.real(np.zeros(1, dtype=dtype)).dtype
    msh = _create_mesh(cell_type, xtype)
    V = fem.functionspace(msh, element)
    u = fem.Function(V, dtype=dtype)
    rng = np.random.default_rng(7)
    u.x.array[:] = rng.random(u.x.array.size)
    if np.issubdtype(dtype, np.complexfloating):
        u.x.array[:] += 1j * rng.random(u.x.array.size)

    x = _interior_points(msh, rng)
    cells = np.arange(x.shape[0], dtype=np.int32)
    tol = _pull_back_tol(xtype)
    basis = fem.evaluate_basis_functions(V, x, cells, tol=tol)
    assert basis.dtype == xtype

    bs = V.dofmap.bs
    coeffs = u.x.array[bs * V.dofmap.list[cells][:, :, None] + np.arange(bs)]
    expanded = np.einsum("pik,piv->pvk", coeffs, basis).reshape(x.shape[0], -1)
    reference = u.eval(x, cells, tol=tol).reshape(x.shape[0], -1)

    # Same pull-back on both paths; only the summation order differs
    eps = np.finfo(xtype).eps
    np.testing.assert_allclose(expanded, reference, rtol=1e3 * eps, atol=1e3 * eps)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_inconsistent_points_and_cells(dtype):
    msh = _create_mesh(mesh.CellType.triangle, dtype)
    V = fem.functionspace(msh, ("Lagrange", 1))
    x = np.zeros((2, 3), dtype=dtype)
    with pytest.raises(ValueError):
        fem.evaluate_basis_functions(V, x, np.zeros(1, dtype=np.int32))


@pytest.mark.parametrize("num_threads", [3, 8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "cell_type, element",
    [
        (mesh.CellType.triangle, ("Lagrange", 2, (2,))),
        (mesh.CellType.quadrilateral, ("Lagrange", 2, (2,))),
        (mesh.CellType.tetrahedron, ("N1curl", 2)),
        (mesh.CellType.hexahedron, ("N1curl", 1)),
    ],
)
def test_threads_match_serial(cell_type, element, dtype, num_threads):
    """Threaded evaluation agrees with serial evaluation.

    Each thread tabulates the reference basis for its own chunk of points,
    and batched tabulation can round differently, so values agree to
    machine precision rather than bitwise.
    """
    msh = _create_mesh(cell_type, dtype)
    V = fem.functionspace(msh, element)
    x = _interior_points(msh, np.random.default_rng(5))
    cells = np.arange(x.shape[0], dtype=np.int32)
    cells[::7] = -1
    tol = _pull_back_tol(dtype)
    basis = fem.evaluate_basis_functions(V, x, cells, tol=tol)
    basis_threaded = fem.evaluate_basis_functions(V, x, cells, tol=tol, num_threads=num_threads)
    atol = 1e2 * np.finfo(dtype).eps * max(np.abs(basis).max(), 1)
    np.testing.assert_allclose(basis_threaded, basis, rtol=0, atol=atol)
    assert np.all(basis_threaded[cells < 0] == 0)

    # More threads than points
    basis_few = fem.evaluate_basis_functions(V, x[:2], cells[:2], tol=tol, num_threads=num_threads)
    np.testing.assert_allclose(basis_few, basis[:2], rtol=0, atol=atol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("cell_type", [mesh.CellType.tetrahedron, mesh.CellType.hexahedron])
def test_function_eval_threads_match_serial(cell_type, dtype):
    xtype = np.real(np.zeros(1, dtype=dtype)).dtype
    msh = _create_mesh(cell_type, xtype)
    V = fem.functionspace(msh, ("N1curl", 2))
    u = fem.Function(V, dtype=dtype)
    rng = np.random.default_rng(11)
    u.x.array[:] = rng.random(u.x.array.size)
    if np.issubdtype(dtype, np.complexfloating):
        u.x.array[:] += 1j * rng.random(u.x.array.size)

    x = _interior_points(msh, rng)
    cells = np.arange(x.shape[0], dtype=np.int32)
    tol = _pull_back_tol(xtype)
    values = u.eval(x, cells, tol=tol)
    values_threaded = u.eval(x, cells, tol=tol, num_threads=4)
    atol = 1e2 * np.finfo(xtype).eps * max(np.abs(values).max(), 1)
    np.testing.assert_allclose(values_threaded, values, rtol=0, atol=atol)


def test_thread_exception_is_raised():
    """A failing pull-back in a worker thread raises, rather than aborting."""
    msh = _create_mesh(mesh.CellType.quadrilateral, np.float64)
    V = fem.functionspace(msh, ("Lagrange", 1))
    x = _interior_points(msh, np.random.default_rng(3))
    cells = np.arange(x.shape[0], dtype=np.int32)

    # With no Newton iterations the non-affine pull-back cannot converge
    with pytest.raises(RuntimeError, match="Newton method failed to converge"):
        fem.evaluate_basis_functions(V, x, cells, maxit=0, num_threads=4)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_blocked_matches_scalar_basis(dtype):
    """A blocked space evaluates the basis of its scalar sub-element.

    The basis is tabulated for one block, so the dof transformation must
    be applied with the scalar element's block size. Serendipity degree
    4 has dof transformations that are not permutations, which makes a
    wrong block size change the values (Lagrange, whose transformations
    are the identity, does not catch this).
    """
    msh = _create_mesh(mesh.CellType.quadrilateral, dtype)
    scalar = basix.ufl.element("Serendipity", "quadrilateral", 4, dtype=dtype)
    V = fem.functionspace(msh, scalar)
    W = fem.functionspace(msh, basix.ufl.blocked_element(scalar, shape=(2,)))

    rng = np.random.default_rng(7)
    x = _interior_points(msh, rng)
    cells = np.arange(x.shape[0], dtype=np.int32)
    tol = _pull_back_tol(dtype)
    blocked = fem.evaluate_basis_functions(W, x, cells, tol=tol)
    reference = fem.evaluate_basis_functions(V, x, cells, tol=tol)
    assert blocked.shape == reference.shape
    np.testing.assert_allclose(blocked, reference)
