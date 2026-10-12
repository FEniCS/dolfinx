# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tests for the interpolation matrix between spaces on different meshes."""

from mpi4py import MPI

import numpy as np
import pytest

import dolfinx
from dolfinx import fem, la, mesh


def _square(n: tuple[int, int], cell_type, dtype, lower=(0.0, 0.0), upper=(1.0, 1.0)):
    return mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array(lower), np.array(upper)],
        n,
        cell_type=cell_type,
        dtype=dtype,
        ghost_mode=mesh.GhostMode.shared_facet,
    )


def _cube(n: tuple[int, int, int], cell_type, dtype):
    return mesh.create_unit_cube(
        MPI.COMM_WORLD, *n, cell_type=cell_type, dtype=dtype, ghost_mode=mesh.GhostMode.shared_facet
    )


def _apply(A: la.MatrixCSR, u0: fem.Function, V1: fem.FunctionSpace) -> fem.Function:
    """Compute A u0 as a function in V1."""
    A.scatter_reverse()
    x = la.vector(A.index_map(1), A.block_size[1], dtype=u0.x.array.dtype)
    imap0 = u0.function_space.dofmap.index_map
    num_owned = imap0.size_local * u0.function_space.dofmap.index_map_bs
    x.array[:num_owned] = u0.x.array[:num_owned]
    x.scatter_forward()
    u1 = fem.Function(V1, dtype=u0.x.array.dtype)
    A.mult(x, u1.x)
    u1.x.scatter_forward()
    return u1


def _reference(u0: fem.Function, V1: fem.FunctionSpace, cells, padding, **kwargs) -> fem.Function:
    """Interpolate u0 into V1 with nonmatching interpolation."""
    u1 = fem.Function(V1, dtype=u0.x.array.dtype)
    data = fem.create_interpolation_data(V1, u0.function_space, cells, padding=padding, **kwargs)
    u1.interpolate_nonmatching(u0, cells, data)
    u1.x.scatter_forward()
    return u1


def _owned_cells(msh: mesh.Mesh) -> np.ndarray:
    return np.arange(msh.topology.index_map(msh.topology.dim).size_local, dtype=np.int32)


def _assert_close(u: fem.Function, u_ref: fem.Function, dtype) -> None:
    num_owned = u.function_space.dofmap.index_map.size_local * u.function_space.dofmap.index_map_bs
    eps = np.finfo(dtype).eps
    scale = max(np.abs(u_ref.x.array).max(initial=0), 1)
    np.testing.assert_allclose(
        u.x.array[:num_owned], u_ref.x.array[:num_owned], rtol=0, atol=1e3 * eps * scale
    )


def _random_function(V: fem.FunctionSpace, dtype) -> fem.Function:
    u = fem.Function(V, dtype=dtype)
    u.x.array[:] = np.random.default_rng(3).random(u.x.array.size)
    u.x.scatter_forward()
    return u


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "dim, element0, element1",
    [
        (2, ("Lagrange", 1), ("Lagrange", 1)),
        (2, ("Lagrange", 2), ("Lagrange", 3)),
        (2, ("Lagrange", 2, (2,)), ("DG", 1, (2,))),
        (2, ("Lagrange", 2, (2,)), ("RT", 1)),
        (2, ("RT", 2), ("RT", 1)),
        (3, ("N1curl", 2), ("N1curl", 2)),
    ],
)
def test_matches_interpolate_nonmatching(dim, element0, element1, dtype):
    if dim == 2:
        mesh0 = _square((5, 7), mesh.CellType.triangle, dtype)
        mesh1 = _square((4, 3), mesh.CellType.quadrilateral, dtype, (0.1, 0.05), (0.9, 0.8))
    else:
        mesh0 = _cube((3, 2, 2), mesh.CellType.tetrahedron, dtype)
        mesh1 = _cube((2, 3, 2), mesh.CellType.tetrahedron, dtype)
    V0 = fem.functionspace(mesh0, element0)
    V1 = fem.functionspace(mesh1, element1)
    u0 = _random_function(V0, dtype)

    eps = np.finfo(dtype).eps
    A = fem.nonmatching_interpolation_matrix(V0, V1, padding=1e2 * eps, tol=np.sqrt(eps))
    u1 = _apply(A, u0, V1)
    u1_ref = _reference(u0, V1, _owned_cells(mesh1), padding=1e2 * eps)
    _assert_close(u1, u1_ref, dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_refined_mesh(dtype):
    """Prolongation from a mesh to its refinement reproduces P2 functions."""
    mesh0 = _square((4, 5), mesh.CellType.triangle, dtype)
    mesh0.topology.create_entities(1)
    mesh1, _, _ = mesh.refine(mesh0)
    V0 = fem.functionspace(mesh0, ("Lagrange", 2))
    V1 = fem.functionspace(mesh1, ("Lagrange", 2))

    def f(x):
        return x[0] ** 2 - 2 * x[0] * x[1] + 3 * x[1]

    u0 = fem.Function(V0, dtype=dtype)
    u0.interpolate(f)
    u1_ref = fem.Function(V1, dtype=dtype)
    u1_ref.interpolate(f)

    eps = np.finfo(dtype).eps
    A = fem.nonmatching_interpolation_matrix(V0, V1, padding=1e2 * eps)
    _assert_close(_apply(A, u0, V1), u1_ref, dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_translation_expansion(dtype):
    """A translated point evaluates the function at the translated point."""
    mesh0 = _square((6, 6), mesh.CellType.triangle, dtype)
    mesh1 = _square((3, 4), mesh.CellType.quadrilateral, dtype, (0.1, 0.1), (0.6, 0.7))
    V0 = fem.functionspace(mesh0, ("Lagrange", 2))
    V1 = fem.functionspace(mesh1, ("Lagrange", 2))
    shift = np.array([0.2, 0.15, 0.0])

    def f(x):
        return x[0] ** 2 + x[0] * x[1] - x[1]

    def translate(x, cells):
        return (x + shift)[:, None, :], np.ones((x.shape[0], 1))

    u0 = fem.Function(V0, dtype=dtype)
    u0.interpolate(f)
    u1_ref = fem.Function(V1, dtype=dtype)
    u1_ref.interpolate(lambda x: f(x + shift[:, None]))

    eps = np.finfo(dtype).eps
    A = fem.nonmatching_interpolation_matrix(V0, V1, expansion=translate, padding=1e2 * eps)
    _assert_close(_apply(A, u0, V1), u1_ref, dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_square_average_expansion(dtype):
    """Weighted points can interpolate an average around each point.

    The 2x2 Gauss rule on a square of half-width h is exact for
    quadratics, and the average of x^2 + 2y^2 over the square is
    x^2 + 2y^2 + h^2.
    """
    mesh0 = _square((6, 6), mesh.CellType.triangle, dtype)
    mesh1 = _square((3, 4), mesh.CellType.triangle, dtype, (0.2, 0.2), (0.8, 0.8))
    V0 = fem.functionspace(mesh0, ("Lagrange", 2))
    V1 = fem.functionspace(mesh1, ("Lagrange", 2))
    h = 0.1
    g = h / np.sqrt(3)
    offsets = np.array([[-g, -g, 0], [g, -g, 0], [-g, g, 0], [g, g, 0]])

    def square_average(x, cells):
        points = x[:, None, :] + offsets[None, :, :]
        return points, np.full((x.shape[0], 4), 0.25)

    u0 = fem.Function(V0, dtype=dtype)
    u0.interpolate(lambda x: x[0] ** 2 + 2 * x[1] ** 2)
    u1_ref = fem.Function(V1, dtype=dtype)
    u1_ref.interpolate(lambda x: x[0] ** 2 + 2 * x[1] ** 2 + h**2)

    eps = np.finfo(dtype).eps
    A = fem.nonmatching_interpolation_matrix(V0, V1, expansion=square_average, padding=1e2 * eps)
    _assert_close(_apply(A, u0, V1), u1_ref, dtype)


def test_unowned_points_do_not_contribute():
    """Points outside the source mesh give zero rows without extrapolation."""
    dtype = dolfinx.default_real_type
    mesh0 = _square((5, 5), mesh.CellType.triangle, dtype)
    mesh1 = _square((4, 4), mesh.CellType.triangle, dtype, (0.5, 0.5), (1.5, 1.5))
    V0 = fem.functionspace(mesh0, ("Lagrange", 1))
    V1 = fem.functionspace(mesh1, ("Lagrange", 1))
    u0 = _random_function(V0, dtype)

    eps = np.finfo(dtype).eps
    A = fem.nonmatching_interpolation_matrix(V0, V1, padding=1e2 * eps, allow_extrapolation=False)
    u1 = _apply(A, u0, V1)
    u1_ref = _reference(u0, V1, _owned_cells(mesh1), padding=1e2 * eps, allow_extrapolation=False)
    _assert_close(u1, u1_ref, dtype)

    # Dofs strictly outside the source mesh are zero
    x1 = V1.tabulate_dof_coordinates()
    num_owned = V1.dofmap.index_map.size_local
    outside = np.any(x1[:num_owned, :2] > 1 + 1e2 * eps, axis=1)
    assert np.all(u1.x.array[:num_owned][outside] == 0)


def test_cell_subset():
    """Only the rows of the given cells are filled."""
    dtype = dolfinx.default_real_type
    mesh0 = _square((5, 5), mesh.CellType.triangle, dtype)
    mesh1 = _square((4, 4), mesh.CellType.quadrilateral, dtype, (0.1, 0.1), (0.9, 0.9))
    V0 = fem.functionspace(mesh0, ("Lagrange", 2))
    V1 = fem.functionspace(mesh1, ("Lagrange", 2))
    u0 = _random_function(V0, dtype)

    cells = mesh.locate_entities(mesh1, 2, lambda x: x[0] < 0.5)
    cells = cells[cells < mesh1.topology.index_map(2).size_local]
    eps = np.finfo(dtype).eps
    A = fem.nonmatching_interpolation_matrix(V0, V1, cells=cells, padding=1e2 * eps)
    u1 = _apply(A, u0, V1)
    u1_ref = _reference(u0, V1, cells, padding=1e2 * eps)
    _assert_close(u1, u1_ref, dtype)


def test_threads_match_serial():
    dtype = dolfinx.default_real_type
    mesh0 = _cube((3, 3, 2), mesh.CellType.hexahedron, dtype)
    mesh1 = _cube((2, 2, 3), mesh.CellType.tetrahedron, dtype)
    V0 = fem.functionspace(mesh0, ("Lagrange", 2))
    V1 = fem.functionspace(mesh1, ("Lagrange", 2))
    u0 = _random_function(V0, dtype)

    eps = np.finfo(dtype).eps
    A = fem.nonmatching_interpolation_matrix(V0, V1, padding=1e2 * eps, tol=np.sqrt(eps))
    A_threaded = fem.nonmatching_interpolation_matrix(
        V0, V1, padding=1e2 * eps, tol=np.sqrt(eps), num_threads=3
    )
    _assert_close(_apply(A_threaded, u0, V1), _apply(A, u0, V1), dtype)


def test_column_map_extends_source_map():
    dtype = dolfinx.default_real_type
    mesh0 = _square((6, 5), mesh.CellType.triangle, dtype)
    mesh1 = _square((5, 6), mesh.CellType.quadrilateral, dtype)
    V0 = fem.functionspace(mesh0, ("Lagrange", 1))
    V1 = fem.functionspace(mesh1, ("Lagrange", 1))
    A = fem.nonmatching_interpolation_matrix(V0, V1)

    imap0 = V0.dofmap.index_map
    column_map = A.index_map(1)
    assert column_map.local_range == imap0.local_range
    ghosts = column_map.ghosts
    assert len(np.unique(ghosts)) == len(ghosts)
    np.testing.assert_array_equal(ghosts[: imap0.num_ghosts], imap0.ghosts)


def test_pull_back_failure_raises_on_all_processes():
    """A failure on one process raises on every process, without a hang."""
    dtype = dolfinx.default_real_type
    mesh0 = _square((4, 4), mesh.CellType.quadrilateral, dtype)
    mesh1 = _square((3, 3), mesh.CellType.triangle, dtype)
    V0 = fem.functionspace(mesh0, ("Lagrange", 1))
    V1 = fem.functionspace(mesh1, ("Lagrange", 1))

    # With no Newton iterations the non-affine pull-back cannot converge
    with pytest.raises(RuntimeError):
        fem.nonmatching_interpolation_matrix(V0, V1, maxit=0)
