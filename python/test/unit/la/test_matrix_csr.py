
import numpy as np
from mpi4py import MPI
from dolfinx.la import matrix_csr
from dolfinx.cpp.la import SparsityPattern, BlockMode
from dolfinx.common import IndexMap


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


def test_add():
    # Regular CSR Matrix 6x6 with bs=1
    sp = create_test_sparsity(6, 1)
    mat1 = matrix_csr(sp)

    # Insert a block using plain indices
    mat1.add([1.0, 2.0, 3.0, 4.0], [2, 3], [4, 5], 1)

    # Insert to same block using bs=2 data
    mat1.add([10.0, 20.0, 30.0, 40.0], [1], [2], 2)

    A1 = mat1.to_dense()

    # Block CSR Matrix 3x3 with bs=2
    sp = create_test_sparsity(3, 2)
    mat2 = matrix_csr(sp)

    # Insert a block using bs=1 data
    mat2.add([10.0, 20.0, 30.0, 40.0], [2, 3], [4, 5], 1)

    # Insert a block using bs=2 data
    mat2.add([1.0, 2.0, 3.0, 4.0], [1], [2], 2)

    A2 = mat2.to_dense()

    assert np.allclose(A1, A2)

    # Block CSR Matrix 3x3 with bs=2, expanded (should be same as A1)
    mat3 = matrix_csr(sp, BlockMode.expanded)

    # Insert a block using bs=1 data
    mat3.add([10.0, 2.0, 30.0, 4.0], [2, 3], [4, 5], 1)

    # Insert a block using bs=2 data
    mat3.add([1.0, 20.0, 3.0, 40.0], [1], [2], 2)

    A3 = mat3.to_dense()
    assert np.allclose(A1, A3)

    mat3.set(0.0)
    assert mat3.squared_norm() == 0.0


def test_set():
    mpi_size = MPI.COMM_WORLD.size
    # Regular CSR Matrix 6x6 with bs=1
    sp = create_test_sparsity(6, 1)
    mat1 = matrix_csr(sp)

    # Set a block with bs=1
    mat1.set([2.0, 3.0, 4.0, 5.0], [2, 3], [4, 5], 1)
    n1 = mat1.squared_norm()
    assert (n1 == 54.0 * mpi_size)

    # Set same block with bs=2
    mat1.set([2.0, 3.0, 4.0, 5.0], [1], [2], 2)
    n2 = mat1.squared_norm()
    assert n1 == n2


def test_set_blocked():
    mpi_size = MPI.COMM_WORLD.size
    # Blocked CSR Matrix 3x3 with bs=2
    sp = create_test_sparsity(3, 2)
    mat1 = matrix_csr(sp)

    # Set a block with bs=1
    mat1.set([2.0, 3.0, 4.0, 5.0], [2, 3], [4, 5], 1)
    n1 = mat1.squared_norm()
    assert (n1 == 54.0 * mpi_size)


def test_distributed_csr():
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
            sp.insert(i, j)
    for i in range(n, n + nghost):
        for j in range(n, n + nghost):
            sp.insert(i, j)
    sp.finalize()

    mat = matrix_csr(sp)
    irow = np.array(range(n), dtype=np.int32)
    icol = np.array(range(n + nghost), dtype=np.int32)
    data = np.ones(len(irow) * len(icol)) * 2.0
    mat.add(data, irow, icol, 1)

    irow = np.array(range(n, n + nghost), dtype=np.int32)
    icol = np.array(range(n, n + nghost), dtype=np.int32)
    data = np.ones(len(irow) * len(icol))
    mat.add(data, irow, icol, 1)
    pre_final_sum = mat.data.sum()
    mat.finalize()
    assert np.isclose(mat.data.sum(), pre_final_sum)
