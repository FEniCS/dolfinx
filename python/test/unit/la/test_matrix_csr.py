# Copyright (C) 2023 Chris N. Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for MatrixCSR"""

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
                sp.insert(np.array([2 + i]), np.array([4 + j]))
    elif bs == 2:
        sp.insert(np.array([1]), np.array([2]))
    sp.finalize()
    return sp


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_add(dtype):
    # Regular CSR Matrix 6x6 with bs=1
    sp = create_test_sparsity(6, 1)
    mat1 = matrix_csr(sp, dtype=dtype)

    # Insert a block using plain indices
    mat1.add([1.0, 2.0, 3.0, 4.0], [2, 3], [4, 5], 1)

    # Insert to same block using bs=2 data
    mat1.add([10.0, 20.0, 30.0, 40.0], [1], [2], 2)

    A1 = mat1.to_dense()

    # Block CSR Matrix 3x3 with bs=2
    sp = create_test_sparsity(3, 2)
    mat2 = matrix_csr(sp, dtype=dtype)

    # Insert a block using bs=1 data
    mat2.add([10.0, 20.0, 30.0, 40.0], [2, 3], [4, 5], 1)

    # Insert a block using bs=2 data
    mat2.add([1.0, 2.0, 3.0, 4.0], [1], [2], 2)

    A2 = mat2.to_dense()

    assert np.allclose(A1, A2)

    # Block CSR Matrix 3x3 with bs=2, expanded (should be same as A1)
    mat3 = matrix_csr(sp, BlockMode.expanded, dtype=dtype)

    # Insert a block using bs=1 data
    mat3.add([10.0, 2.0, 30.0, 4.0], [2, 3], [4, 5], 1)

    # Insert a block using bs=2 data
    mat3.add([1.0, 20.0, 3.0, 40.0], [1], [2], 2)

    A3 = mat3.to_dense()
    assert np.allclose(A1, A3)

    mat3.set_value(0.0)
    assert mat3.squared_norm() == 0.0  # /NOSONAR


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_set(dtype):
    mpi_size = MPI.COMM_WORLD.size
    # Regular CSR Matrix 6x6 with bs=1
    sp = create_test_sparsity(6, 1)
    mat1 = matrix_csr(sp, dtype=dtype)

    # Set a block with bs=1
    mat1.set([2.0, 3.0, 4.0, 5.0], [2, 3], [4, 5], 1)
    n1 = mat1.squared_norm()
    assert n1 == 54.0 * mpi_size  # /NOSONAR

    # Set same block with bs=2
    mat1.set([2.0, 3.0, 4.0, 5.0], [1], [2], 2)
    n2 = mat1.squared_norm()
    assert n1 == n2


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_set_blocked(dtype):
    mpi_size = MPI.COMM_WORLD.size
    # Blocked CSR Matrix 3x3 with bs=2
    sp = create_test_sparsity(3, 2)
    mat1 = matrix_csr(sp, dtype=dtype)

    # Set a block with bs=1
    mat1.set([2.0, 3.0, 4.0, 5.0], [2, 3], [4, 5], 1)
    n1 = mat1.squared_norm()
    assert n1 == 54.0 * mpi_size  # /NOSONAR


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_distributed_csr(dtype):
    # global size N
    N = 36
    nghost = 3
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    nbr = (rank + 1) % size
    n = int(N / size)
    ghosts = np.array(range(n * nbr, n * nbr + nghost), dtype=np.int64)
    owner = np.ones_like(ghosts, dtype=np.int32) * nbr

    im = IndexMap(MPI.COMM_WORLD, n, ghosts, owner)
    sp = SparsityPattern(MPI.COMM_WORLD, [im, im], [1, 1])
    for i in range(n):
        for j in range(n + nghost):
            sp.insert(np.array([i]), np.array([j]))
    for i in range(n, n + nghost):
        for j in range(n, n + nghost):
            sp.insert(np.array([i]), np.array([j]))
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


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
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


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
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

    shared_dofs = index_map.index_to_dest_ranks()
    for dof in range(nlocal):
        owners = shared_dofs.links(dof)
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
        mat1.set([1.0, 2.0, 3.0, 4.0], [0], [0], 2)

    # Set an single entry in bs=1 matrix (tests insert_csr)
    with pytest.raises(RuntimeError):
        mat1.add([1.0], [0], [0], 1)

    sp = create_test_sparsity(3, 2)
    mat2 = matrix_csr(sp, BlockMode.compact, dtype=dtype)
    # set unblocked in bs=2 matrix (tests insert_nonblocked_csr)
    with pytest.raises(RuntimeError):
        mat2.add([2.0, 3.0, 4.0, 5.0], [0, 1], [0, 1], 1)
