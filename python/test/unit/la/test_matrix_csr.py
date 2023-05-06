
import numpy as np
from mpi4py import MPI
from dolfinx.la import matrix_csr
from dolfinx.cpp.la import SparsityPattern, BlockMode
from dolfinx.common import IndexMap


def test_add():

    # Regular CSR Matrix 6x6 with bs=1
    n = 6
    im = IndexMap(MPI.COMM_WORLD, n)
    sp = SparsityPattern(MPI.COMM_WORLD, [im, im], [1, 1])
    for i in range(2):
        for j in range(2):
            sp.insert(2 + i, 4 + j)
    sp.assemble()

    mat1 = matrix_csr(sp)

    # Insert a block
    mat1.add([1.0, 2.0, 3.0, 4.0], [2, 3], [4, 5], 1)

    # Insert a block using bs=2 data
    mat1.add([10.0, 20.0, 30.0, 40.0], [1], [2], 2)

    A1 = mat1.to_dense()

    # Block CSR Matrix 3x3 with bs=2
    n = 3
    im = IndexMap(MPI.COMM_WORLD, n)
    sp = SparsityPattern(MPI.COMM_WORLD, [im, im], [2, 2])
    sp.insert(1, 2)
    sp.assemble()
    mat2 = matrix_csr(sp)

    # Insert a block using bs=1 data
    mat2.add([10.0, 20.0, 30.0, 40.0], [2, 3], [4, 5], 1)

    # Insert a block using bs=2 data
    mat2.add([1.0, 2.0, 3.0, 4.0], [1], [2], 2)

    A2 = mat2.to_dense()

    assert np.allclose(A1, A2)

    # Block CSR Matrix 3x3 with bs=2, expanded (should be same as A1)
    n = 3
    im = IndexMap(MPI.COMM_WORLD, n)
    sp = SparsityPattern(MPI.COMM_WORLD, [im, im], [2, 2])
    sp.insert(1, 2)
    sp.assemble()
    mat3 = matrix_csr(sp, BlockMode.expanded)

    # Insert a block using bs=1 data
    mat3.add([10.0, 2.0, 30.0, 4.0], [2, 3], [4, 5], 1)

    # Insert a block using bs=2 data
    mat3.add([1.0, 20.0, 3.0, 40.0], [1], [2], 2)

    A3 = mat3.to_dense()
    assert np.allclose(A1, A3)

    mat3.set(0.0)
    assert(mat3.squared_norm() == 0.0)

    mat3.set([2.0, 3.0, 4.0, 5.0], [1], [2], 2)
    n1 = mat3.squared_norm()
    mat3.set([2.0, 3.0, 4.0, 5.0], [1], [2], 2)
    n2 = mat3.squared_norm()
    assert n1 == n2
