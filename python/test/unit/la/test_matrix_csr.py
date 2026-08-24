# Copyright (C) 2023 Chris N. Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for MatrixCSR."""

from mpi4py import MPI

import numpy as np
import pytest

import ufl
from dolfinx import cpp as _cpp
from dolfinx import fem
from dolfinx.common import IndexMap
from dolfinx.cpp.la import BlockMode, SparsityPattern
from dolfinx.la import matrix_csr
from dolfinx.mesh import GhostMode, create_unit_square


def create_test_sparsity(n, bs):
    im = IndexMap(MPI.COMM_WORLD, n)
    sp = SparsityPattern(MPI.COMM_WORLD, [im, im], [bs, bs])
    if bs == 1:
        for i in range(2):
            for j in range(2):
                sp.insert(2 + i, 4 + j)
    elif bs == 2:
        sp.insert(1, 2)
    sp.finalize()
    return sp


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_add(dtype):
    # Regular CSR Matrix 6x6 with bs=1
    sp = create_test_sparsity(6, 1)
    mat1 = matrix_csr(sp, dtype=dtype)

    # Insert a block using plain indices
    mat1.add(np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype), np.array([2, 3]), np.array([4, 5]), 1)

    # Insert to same block using bs=2 data
    mat1.add(np.array([10.0, 20.0, 30.0, 40.0], dtype=dtype), np.array([1]), np.array([2]), 2)

    A1 = mat1.to_dense()

    # Block CSR Matrix 3x3 with bs=2
    sp = create_test_sparsity(3, 2)
    mat2 = matrix_csr(sp, dtype=dtype)

    # Insert a block using bs=1 data
    mat2.add(np.array([10.0, 20.0, 30.0, 40.0], dtype=dtype), np.array([2, 3]), np.array([4, 5]), 1)

    # Insert a block using bs=2 data
    mat2.add(np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype), np.array([1]), np.array([2]), 2)

    A2 = mat2.to_dense()

    assert np.allclose(A1, A2)

    # Block CSR Matrix 3x3 with bs=2, expanded (should be same as A1)
    mat3 = matrix_csr(sp, BlockMode.expanded, dtype=dtype)

    # Insert a block using bs=1 data
    mat3.add(np.array([10.0, 2.0, 30.0, 4.0]), np.array([2, 3]), np.array([4, 5]), 1)

    # Insert a block using bs=2 data
    mat3.add(np.array([1.0, 20.0, 3.0, 40.0]), np.array([1]), np.array([2]), 2)

    A3 = mat3.to_dense()
    assert np.allclose(A1, A3)

    mat3.set_value(0)
    assert mat3.squared_norm() == 0.0  # /NOSONAR


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_set(dtype):
    mpi_size = MPI.COMM_WORLD.size
    # Regular CSR Matrix 6x6 with bs=1
    sp = create_test_sparsity(6, 1)
    mat1 = matrix_csr(sp, dtype=dtype)

    # Set a block with bs=1
    mat1.set(np.array([2.0, 3.0, 4.0, 5.0], dtype=dtype), np.array([2, 3]), np.array([4, 5]), 1)
    n1 = mat1.squared_norm()
    assert n1 == 54.0 * mpi_size  # /NOSONAR

    # Set same block with bs=2
    mat1.set(np.array([2.0, 3.0, 4.0, 5.0], dtype=dtype), np.array([1]), np.array([2]), 2)
    n2 = mat1.squared_norm()
    assert n1 == n2


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_set_blocked(dtype):
    mpi_size = MPI.COMM_WORLD.size
    # Blocked CSR Matrix 3x3 with bs=2
    sp = create_test_sparsity(3, 2)
    mat1 = matrix_csr(sp, dtype=dtype)

    # Set a block with bs=1
    mat1.set(np.array([2.0, 3.0, 4.0, 5.0], dtype=dtype), np.array([2, 3]), np.array([4, 5]), 1)
    n1 = mat1.squared_norm()
    assert n1 == 54.0 * mpi_size  # /NOSONAR


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_distributed_csr(dtype):
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    if size == 1:
        return

    # global size N
    N = 36
    nghost = 3
    nbr = (rank + 1) % size
    n = int(N / size)
    ghosts = np.array(range(n * nbr, n * nbr + nghost), dtype=np.int64)
    owner = np.ones_like(ghosts, dtype=np.int32) * nbr

    im = IndexMap(MPI.COMM_WORLD, n, ghosts, owner, 0)
    sp = SparsityPattern(MPI.COMM_WORLD, [im, im], [1, 1])
    for i in range(n):
        for j in range(n + nghost):
            sp.insert(i, j)
    for i in range(n, n + nghost):
        for j in range(n, n + nghost):
            sp.insert(i, j)
    sp.finalize()

    mat = matrix_csr(sp, dtype=dtype)
    irow = np.array(range(n), dtype=np.int32)
    icol = np.array(range(n + nghost), dtype=np.int32)
    data = np.ones(len(irow) * len(icol), dtype=dtype) * 2.0
    mat.add(data, irow, icol, 1)

    irow = np.array(range(n, n + nghost), dtype=np.int32)
    icol = np.array(range(n, n + nghost), dtype=np.int32)
    data = np.ones(len(irow) * len(icol), dtype=dtype)
    mat.add(data, irow, icol, 1)
    pre_final_sum = mat.data.sum()
    mat.scatter_reverse()
    assert np.isclose(mat.data.sum(), pre_final_sum)


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        pytest.param(np.complex64, marks=pytest.mark.xfail_win32_complex),
        pytest.param(np.complex128, marks=pytest.mark.xfail_win32_complex),
    ],
)
def test_set_block_matrix(dtype):
    mesh_dtype = np.real(dtype(0)).dtype
    ghost_mode = GhostMode.shared_facet
    mesh = create_unit_square(MPI.COMM_WORLD, 2, 4, ghost_mode=ghost_mode, dtype=mesh_dtype)
    V = fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = fem.form(ufl.inner(u, v) * ufl.dx, dtype=dtype)
    A = fem.create_matrix(a)
    As = A.to_scipy(ghosted=True)
    assert As.blocksize == (2, 2)


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        pytest.param(np.complex64, marks=pytest.mark.xfail_win32_complex),
        pytest.param(np.complex128, marks=pytest.mark.xfail_win32_complex),
    ],
)
def test_set_diagonal_distributed(dtype):
    mesh_dtype = np.real(dtype(0)).dtype
    ghost_mode = GhostMode.shared_facet
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, ghost_mode=ghost_mode, dtype=mesh_dtype)
    V = fem.functionspace(mesh, ("Lagrange", 1))

    tdim = mesh.topology.dim
    cellmap = mesh.topology.index_map(tdim)
    num_cells = cellmap.size_local + cellmap.num_ghosts

    # Integration domain includes ghost cells
    cells_domains = [(1, np.arange(0, num_cells))]
    dx = ufl.Measure("dx", subdomain_data=cells_domains, domain=mesh)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = fem.form(ufl.inner(u, v) * dx(1), dtype=dtype)

    # get index map from function space
    index_map = V.dofmap.index_map
    num_dofs = index_map.size_local + index_map.num_ghosts

    # list of dofs including ghost dofs
    dofs = np.arange(0, num_dofs, dtype=np.int32)

    # create matrix
    A = fem.create_matrix(a)
    As = A.to_scipy(ghosted=True)

    # set diagonal values
    value = dtype(1.0)
    _cpp.fem.insert_diagonal(A._cpp_object, dofs, value)

    # check diagonal values: they should be 1.0, including ghost dofs
    diag = As.diagonal()
    reference = np.full_like(diag, value, dtype=dtype)
    assert np.allclose(diag, reference)

    # Update matrix: this will remove ghost rows and diagonal values of
    # ghost rows will be added to diagonal of corresponding process
    A.scatter_reverse()

    diag = As.diagonal()
    nlocal = index_map.size_local
    assert (diag[nlocal:] == dtype(0.0)).all()

    data, offsets = index_map.index_to_dest_ranks(0)
    for dof in range(nlocal):
        owners = data[offsets[dof] : offsets[dof + 1]]
        assert diag[dof] == len(owners) + 1

    # create matrix
    A = fem.create_matrix(a)
    As = A.to_scipy(ghosted=True)

    # set diagonal values using dirichlet bc: this will set diagonal values of
    # owned rows only
    bc = fem.dirichletbc(dtype(0.0), dofs, V)
    _cpp.fem.insert_diagonal(A._cpp_object, a.function_spaces[0], [bc._cpp_object], value)

    # check diagonal values: they should be 1.0, except ghost dofs
    diag = As.diagonal()
    reference = np.full_like(diag, value, dtype=dtype)
    assert np.allclose(diag[:nlocal], reference[:nlocal])
    assert np.allclose(diag[nlocal:], np.zeros_like(diag[nlocal:]))

    # Update matrix:
    # this will zero ghost rows and diagonal values are already zero.
    A.scatter_reverse()
    assert (As.diagonal()[nlocal:] == dtype(0.0)).all()
    assert (As.diagonal()[:nlocal] == dtype(1.0)).all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_bad_entry(dtype):
    sp = create_test_sparsity(6, 1)
    mat1 = matrix_csr(sp, dtype=dtype)

    # Set block in bs=1 matrix (tests insert_blocked_csr)
    with pytest.raises(RuntimeError):
        mat1.set(np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype), np.array([0]), np.array([0]), 2)

    # Set an single entry in bs=1 matrix (tests insert_csr)
    with pytest.raises(RuntimeError):
        mat1.add(np.array([1.0], dtype=dtype), np.array([0]), np.array([0]), 1)

    sp = create_test_sparsity(3, 2)
    mat2 = matrix_csr(sp, BlockMode.compact, dtype=dtype)
    # set unblocked in bs=2 matrix (tests insert_nonblocked_csr)
    with pytest.raises(RuntimeError):
        mat2.add(np.array([2.0, 3.0, 4.0, 5.0], dtype=dtype), np.array([0, 1]), np.array([0, 1]), 1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_eliminate_zeros_tolerance(dtype):
    """Entries are removed from storage iff |value| <= tol (i.e. kept iff
    strictly greater than tol), and storage is compacted accordingly.
    """
    sp = create_test_sparsity(6, 1)
    mat = matrix_csr(sp, dtype=dtype)

    # ``to_dense()`` indexes columns *globally*, so on more than one rank
    # this rank's local columns 4 and 5 are offset by its column range
    # start -- only rank 0 has that offset equal to zero.
    col0 = mat.index_map(1).local_range[0]

    # The sparsity pattern from create_test_sparsity(6, 1) has exactly four
    # explicit entries: (2, 4), (2, 5), (3, 4), (3, 5). Populate them with
    # values straddling a tolerance of 1.0: an exact zero, a value below
    # tol, a value exactly at the tol boundary, and a value clearly above.
    below, boundary, above = dtype(0.5), dtype(1.0), dtype(2.0)
    mat.add(np.array([0.0, below, boundary, above], dtype=dtype), np.array([1]), np.array([2]), 2)

    A_before = mat.to_dense()
    nnz_before = int(mat.indptr[-1])
    assert nnz_before == 4

    tol = dtype(1.0)
    mat.eliminate_zeros(tol)

    nnz_after = int(mat.indptr[-1])
    A_after = mat.to_dense()

    # Only the entry strictly greater in magnitude than tol should survive:
    # the exact zero, the below-tolerance value, and the boundary value
    # (== tol, not > tol) must all be dropped from storage.
    assert nnz_after == 1
    assert A_after[2, col0 + 4] == 0
    assert A_after[2, col0 + 5] == 0
    assert A_after[3, col0 + 4] == 0
    assert np.isclose(A_after[3, col0 + 5], above)

    # Values above tolerance are left completely untouched.
    assert A_after[3, col0 + 5] == A_before[3, col0 + 5]


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_eliminate_zeros_default_tolerance(dtype):
    """With no tolerance supplied, only exact structural zeros are removed;
    small-but-nonzero entries must survive.

    Note:
        The ``dolfinx.la.MatrixCSR.eliminate_zeros`` Python wrapper always
        forwards an explicit value to the C++ layer, so this default is
        applied in Python rather than relying on the nanobind binding's
        own default argument.
    """
    sp = create_test_sparsity(6, 1)
    mat = matrix_csr(sp, dtype=dtype)

    # ``to_dense()`` indexes columns *globally*, so on more than one rank
    # this rank's local columns 4 and 5 are offset by its column range
    # start -- only rank 0 has that offset equal to zero.
    col0 = mat.index_map(1).local_range[0]

    tiny = dtype(1e-6)
    mat.add(np.array([0.0, tiny, 10.0, 40.0], dtype=dtype), np.array([1]), np.array([2]), 2)

    assert int(mat.indptr[-1]) == 4

    mat.eliminate_zeros()  # default tol = 0

    assert int(mat.indptr[-1]) == 3  # only the exact zero is dropped

    A = mat.to_dense()
    assert A[2, col0 + 4] == 0
    assert np.isclose(A[2, col0 + 5], tiny)
    assert np.isclose(A[3, col0 + 4], 10.0)
    assert np.isclose(A[3, col0 + 5], 40.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_eliminate_zeros_no_change_when_nothing_within_tolerance(dtype):
    """eliminate_zeros must be a no-op (and not corrupt data) when no
    entries fall within the given tolerance.
    """
    sp = create_test_sparsity(6, 1)
    mat = matrix_csr(sp, dtype=dtype)
    mat.add(np.array([10.0, 20.0, 30.0, 40.0], dtype=dtype), np.array([1]), np.array([2]), 2)

    nnz_before = int(mat.indptr[-1])
    A_before = mat.to_dense()

    mat.eliminate_zeros(dtype(1.0))

    assert int(mat.indptr[-1]) == nnz_before
    assert np.allclose(mat.to_dense(), A_before)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_eliminate_zeros_blocked_partial(dtype):
    """A block with any entry above tolerance must be kept in full, even
    though some of its other entries are within tolerance.
    """
    sp = create_test_sparsity(3, 2)
    mat = matrix_csr(sp, BlockMode.compact, dtype=dtype)

    # Single bs=2 block at block-row 1, block-col 2 -> rows (2, 3), cols (4, 5).
    # Not all entries are within tolerance, so the whole block must survive
    # unchanged.
    mat.add(np.array([0.0, 0.5, 1.0, 5.0], dtype=dtype), np.array([1]), np.array([2]), 2)
    A_before = mat.to_dense()

    mat.eliminate_zeros(dtype(1.0))

    assert np.allclose(mat.to_dense(), A_before)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_eliminate_zeros_blocked_whole_block(dtype):
    """A block is only dropped from storage when *every* one of its
    bs0*bs1 entries is within tolerance; a block with even one entry
    above tolerance is kept in full, byte-for-byte.
    """
    im = IndexMap(MPI.COMM_WORLD, 4)
    sp = SparsityPattern(MPI.COMM_WORLD, [im, im], [2, 2])
    sp.insert(0, 1)
    sp.insert(2, 3)
    sp.finalize()
    mat = matrix_csr(sp, BlockMode.compact, dtype=dtype)

    # Block at block-row 0, block-col 1 -> rows (0, 1), cols (2, 3).
    # Every entry is within tolerance, so the whole block should be
    # dropped from storage.
    mat.add(np.array([0.1, 0.2, 0.3, 0.4], dtype=dtype), np.array([0]), np.array([1]), 2)
    # Block at block-row 2, block-col 3 -> rows (4, 5), cols (6, 7). One
    # entry exceeds tolerance, so the whole block must be kept unchanged.
    mat.add(np.array([0.1, 0.2, 0.3, 5.0], dtype=dtype), np.array([2]), np.array([3]), 2)

    A_before = mat.to_dense()
    assert int(mat.indptr[-1]) == 2  # two blocks in storage

    mat.eliminate_zeros(dtype(1.0))

    assert int(mat.indptr[-1]) == 1  # the all-below-tolerance block is dropped

    A_after = mat.to_dense()
    assert np.allclose(A_after[0:2, 2:4], 0)
    assert np.array_equal(A_after[4:6, 6:8], A_before[4:6, 6:8])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_eliminate_zeros_finalizes(dtype):
    """eliminate_zeros() can shrink the sparsity, which invalidates the
    precomputed ghost-row communication pattern. Once called, further
    modification of the matrix must be rejected rather than silently
    corrupting data.
    """
    sp = create_test_sparsity(6, 1)
    mat = matrix_csr(sp, dtype=dtype)
    mat.add(np.array([10.0, 20.0, 30.0, 40.0], dtype=dtype), np.array([1]), np.array([2]), 2)

    mat.eliminate_zeros(dtype(1.0))

    with pytest.raises(RuntimeError):
        mat.add(np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype), np.array([1]), np.array([2]), 2)

    with pytest.raises(RuntimeError):
        mat.set(np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype), np.array([1]), np.array([2]), 2)

    with pytest.raises(RuntimeError):
        mat.scatter_reverse()
