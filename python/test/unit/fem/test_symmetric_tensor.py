# Copyright (C) 2023-2026 Matthew W. Scroggs and Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
import sys

from mpi4py import MPI

import numpy as np
import pytest

import basix.ufl
import dolfinx
import ufl


@pytest.mark.parametrize("degree", range(1, 4))
@pytest.mark.parametrize("symmetry", [True, False])
def test_transpose(degree, symmetry) -> None:
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    e = basix.ufl.element(
        "Lagrange",
        "triangle",
        degree,
        shape=(2, 2),
        symmetry=symmetry,
        dtype=dolfinx.default_real_type,
    )
    space = dolfinx.fem.functionspace(mesh, e)
    f = dolfinx.fem.Function(space)
    f.interpolate(lambda x: [x[0], x[1], 2 * x[1], x[0] ** 3])

    form = dolfinx.fem.form(ufl.inner(f - ufl.transpose(f), f - ufl.transpose(f)) * ufl.dx)
    assert np.isclose(dolfinx.fem.assemble_scalar(form), 0) == symmetry


@pytest.mark.parametrize("n", [2, 3])
def test_interpolation_nonsymmetric_data(n) -> None:
    """Non-symmetric input is accepted and produces a symmetric tensor.

    The diagonal of the result is the diagonal of the input; which
    off-diagonal components are retained is unspecified, so it is not
    checked.
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    mat = np.arange(1, n * n + 1, dtype=dolfinx.default_scalar_type).reshape(n, n)

    def tensor(x):
        return np.broadcast_to(mat.reshape(n * n, 1), (n * n, x.shape[1]))

    def space(symmetry):
        e = basix.ufl.element(
            "DG",
            mesh.basix_cell(),
            0,
            shape=(n, n),
            symmetry=symmetry,
            dtype=dolfinx.default_real_type,
        )
        return dolfinx.fem.functionspace(mesh, e)

    V = space(True)
    cells = np.arange(mesh.topology.index_map(2).size_local, dtype=np.int32)
    points = dolfinx.mesh.compute_midpoints(mesh, 2, cells)
    atol = 10 * np.finfo(dolfinx.default_scalar_type).resolution

    # Callable source
    f = dolfinx.fem.Function(V)
    f.interpolate(tensor)
    values = f.eval(points, cells).reshape(-1, n, n)
    assert np.allclose(values, values.swapaxes(1, 2), atol=atol)
    assert np.allclose(np.diagonal(values, axis1=1, axis2=2), np.diag(mat), atol=atol)

    # Expression source, built on a non-symmetric Function
    g = dolfinx.fem.Function(space(False))
    g.interpolate(tensor)
    f.x.array[:] = 0
    f.interpolate(dolfinx.fem.Expression(2 * g, V.element.interpolation_points))
    values = f.eval(points, cells).reshape(-1, n, n)
    assert np.allclose(values, values.swapaxes(1, 2), atol=atol)
    assert np.allclose(np.diagonal(values, axis1=1, axis2=2), 2 * np.diag(mat), atol=atol)

    # Direct Function source
    f.x.array[:] = 0
    f.interpolate(g)
    values = f.eval(points, cells).reshape(-1, n, n)
    assert np.allclose(values, values.swapaxes(1, 2), atol=atol)
    assert np.allclose(np.diagonal(values, axis1=1, axis2=2), np.diag(mat), atol=atol)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("family", ["Lagrange", "DG"])
@pytest.mark.parametrize("degrees", [(1, 1), (1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize("source_symmetry", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_interpolation_between_tensor_layouts(dim, family, degrees, source_symmetry, dtype):
    """Interpolate exact tensor values between full and independent component layouts."""
    real_type = dtype(0).real.dtype
    if dim == 2:
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2, dtype=real_type)
    else:
        mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, dtype=real_type)

    def tensor(x):
        points = x[:dim, None] ** min(degrees)
        values = points + points.swapaxes(0, 1) + np.eye(dim)[:, :, None]
        if np.issubdtype(dtype, np.complexfloating):
            values = (1 + 2j) * values
        return values.reshape(dim * dim, -1)

    functions = []
    for degree, symmetry in zip(degrees, [source_symmetry, not source_symmetry], strict=True):
        element = basix.ufl.element(
            family, mesh.basix_cell(), degree, shape=(dim, dim), symmetry=symmetry, dtype=real_type
        )
        functions.append(
            dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, element), dtype=dtype)
        )
    source, target = functions
    source.interpolate(tensor)
    target.interpolate(source)

    cells = np.arange(mesh.topology.index_map(dim).size_local, dtype=np.int32)
    points = dolfinx.mesh.compute_midpoints(mesh, dim, cells)
    values = target.eval(points, cells)
    expected = tensor(points.T).T
    atol = 100 * np.finfo(real_type).eps
    np.testing.assert_allclose(values, expected, rtol=atol, atol=atol)


@pytest.mark.parametrize("source_symmetry", [True, False])
def test_interpolation_tensor_layouts_cell_selection(source_symmetry):
    """Convert tensor layouts on a subset of cells with reordered source cells."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    functions = []
    for symmetry in [source_symmetry, not source_symmetry]:
        element = basix.ufl.element(
            "DG",
            mesh.basix_cell(),
            0,
            shape=(2, 2),
            symmetry=symmetry,
            dtype=dolfinx.default_real_type,
        )
        functions.append(dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, element)))
    source, target = functions
    source.interpolate(lambda x: [x[0], x[1], x[1], 2 * x[0]])
    target.x.array[:] = np.nan

    all_cells = np.arange(mesh.topology.index_map(2).size_local, dtype=np.int32)
    cells = all_cells[::2].copy()
    # `cells0[i]` and `cells1[i]` are documented to be the same cell, so on a
    # single mesh they would normally be equal. Reversing `cells0` only checks
    # that the two lists are not transposed internally, and is safe here solely
    # because DG0 is identity mapped: the conversion path takes the cell
    # geometry from `cells0` while writing the dofs of `cells1`.
    source_cells = cells[::-1].copy()
    target.interpolate(source, cells0=source_cells, cells1=cells)
    points = dolfinx.mesh.compute_midpoints(mesh, 2, cells)
    source_points = dolfinx.mesh.compute_midpoints(mesh, 2, source_cells)
    atol = 100 * np.finfo(dolfinx.default_real_type).eps
    np.testing.assert_allclose(
        target.eval(points, cells), source.eval(source_points, source_cells), rtol=atol, atol=atol
    )
    untouched_cells = all_cells[1::2].copy()
    untouched_points = dolfinx.mesh.compute_midpoints(mesh, 2, untouched_cells)
    assert np.isnan(target.eval(untouched_points, untouched_cells)).all()


def test_interpolation() -> None:
    """Test that a symmetric 3x3 2-tensor is correctly interpolated."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    def tensor(x):
        mat = np.array([[0], [1], [2], [1], [3], [4], [2], [4], [5]])
        return np.broadcast_to(mat, (9, x.shape[1]))

    element = basix.ufl.element(
        "DG", mesh.basix_cell(), 0, shape=(3, 3), symmetry=False, dtype=dolfinx.default_real_type
    )
    space = dolfinx.fem.functionspace(mesh, element)
    f = dolfinx.fem.Function(space)
    f.interpolate(lambda x: tensor(x))

    symm_element = basix.ufl.element(
        "DG", mesh.basix_cell(), 0, shape=(3, 3), symmetry=True, dtype=dolfinx.default_real_type
    )
    symm_space = dolfinx.fem.functionspace(mesh, symm_element)
    symm_f = dolfinx.fem.Function(symm_space)
    symm_f.interpolate(lambda x: tensor(x))

    l2_error = dolfinx.fem.assemble_scalar(dolfinx.fem.form((f - symm_f) ** 2 * ufl.dx))
    atol = 10 * np.finfo(dolfinx.default_scalar_type).resolution
    assert np.isclose(l2_error, 0.0, atol=atol)

    # Only the stored components: the lower triangle, row by row
    def stored_tensor(x):
        mat = np.array([[0], [1], [3], [2], [4], [5]])
        return np.broadcast_to(mat, (6, x.shape[1]))

    symm_f.x.array[:] = 0
    symm_f.interpolate(stored_tensor)
    l2_error = dolfinx.fem.assemble_scalar(dolfinx.fem.form((f - symm_f) ** 2 * ufl.dx))
    assert np.isclose(l2_error, 0.0, atol=atol)


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("family, degree", [("DG", 0), ("Lagrange", 2)])
def test_interpolation_stored_components(n, family, degree) -> None:
    """A callable may return only the n * (n + 1) / 2 stored components.

    They are the lower triangle, row by row, and give the same result as
    returning all n * n components.
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    lower = [(i, j) for i in range(n) for j in range(i + 1)]

    def stored(x):
        return np.stack([(k + 1) * x[0] + x[1] ** 2 - k * x[0] * x[1] for k in range(len(lower))])

    def full(x):
        s = stored(x)
        return np.stack(
            [s[lower.index((max(i, j), min(i, j)))] for i in range(n) for j in range(n)]
        )

    def space(symmetry):
        e = basix.ufl.element(
            family,
            mesh.basix_cell(),
            degree,
            shape=(n, n),
            symmetry=symmetry,
            dtype=dolfinx.default_real_type,
        )
        return dolfinx.fem.functionspace(mesh, e)

    V = space(True)
    f_full = dolfinx.fem.Function(V)
    f_full.interpolate(full)
    f_stored = dolfinx.fem.Function(V)
    f_stored.interpolate(stored)
    assert np.any(f_full.x.array != 0)
    assert np.array_equal(f_stored.x.array, f_full.x.array)

    # Other sizes are rejected
    with pytest.raises(ValueError, match="wrong shape"):
        f_stored.interpolate(lambda x: stored(x)[:-1])

    # A non-symmetric space still requires all n * n components
    g = dolfinx.fem.Function(space(False))
    with pytest.raises(ValueError, match="wrong shape"):
        g.interpolate(stored)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("symmetry", [True, False])
@pytest.mark.parametrize("etype", ["Regge", "HHJ"])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        pytest.param(
            np.complex64,
            marks=pytest.mark.xfail(
                sys.platform.startswith("win32"),
                raises=NotImplementedError,
                reason="missing _Complex",
            ),
        ),
        pytest.param(
            np.complex128,
            marks=pytest.mark.xfail(
                sys.platform.startswith("win32"),
                raises=NotImplementedError,
                reason="missing _Complex",
            ),
        ),
    ],
)
def test_interpolation_symmetric_tensor(dim, symmetry, etype, degree, dtype) -> None:
    """Tests the interpolation path for non-matching maps of symmetric elements."""
    comm = MPI.COMM_WORLD
    real_type = dtype(0).real.dtype
    if dim == 2:
        mesh = dolfinx.mesh.create_unit_square(comm, 5, 5, dtype=real_type)
    else:
        mesh = dolfinx.mesh.create_unit_cube(comm, 5, 5, 5, dtype=real_type)

    def tensor(x):
        # Create symmetric linear tensor which is contained in all matrix element spaces exactly and
        # thus perfectly reproducible by interpolation.
        #   a_ij = a_i + a_j + δ_ij
        points = x[:dim, None]
        A = points + points.swapaxes(0, 1) + np.eye(dim)[:, :, None]
        return A.reshape(dim * dim, -1)

    matrix_element = basix.ufl.element(
        etype,
        mesh.basix_cell(),
        1,
        dtype=real_type,
    )
    u_matrix = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, matrix_element), dtype=dtype)

    element = basix.ufl.element(
        "Lagrange",
        mesh.basix_cell(),
        degree,
        shape=(dim, dim),
        symmetry=symmetry,
        dtype=real_type,
    )
    u_lagrange = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, element), dtype=dtype)

    def check_norm():
        l2_error = comm.allreduce(
            dolfinx.fem.assemble_scalar(
                dolfinx.fem.form((u_lagrange - u_matrix) ** 2 * ufl.dx, dtype=dtype)
            )
        )
        assert np.isclose(l2_error, 0.0, atol=10 * np.finfo(dtype).resolution)

    # Tensor to (symmetric) Lagrange
    u_matrix.interpolate(tensor)
    u_lagrange.interpolate(u_matrix)
    check_norm()

    # (symmetric) Lagrange to Tensor
    u_lagrange.interpolate(tensor)
    u_matrix.x.array[:] = 0
    u_matrix.interpolate(u_lagrange)
    check_norm()


def test_eval():
    """Test that eval is correct for a symmetric 3x3 2-tensor is correct."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    mat = np.array([0, 1, 2, 1, 3, 4, 2, 4, 5])

    def tensor(x):
        return np.broadcast_to(mat.reshape((9, 1)), (9, x.shape[1]))

    element = basix.ufl.element(
        "DG", mesh.basix_cell(), 0, shape=(3, 3), symmetry=True, dtype=dolfinx.default_real_type
    )
    space = dolfinx.fem.functionspace(mesh, element)
    f = dolfinx.fem.Function(space)
    f.interpolate(lambda x: tensor(x))
    value = f.eval([[0, 0, 0]], [0])
    atol = 10 * np.finfo(dolfinx.default_scalar_type).resolution
    assert np.allclose(value, mat, atol=atol)
