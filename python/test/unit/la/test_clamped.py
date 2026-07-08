"""Test clamping of negligible matrix entries to zero on insertion.

The insertion functors ``MatrixCSR::mat_set_values_clamp`` and
``MatrixCSR::mat_add_values_clamp`` set entries that are negligible
relative to the largest-magnitude entry of the inserted block to
exactly zero: an entry ``a`` is zeroed when
``|a| < C * eps * max(|block|)``, where ``eps`` is the machine epsilon
of the real scalar type. This removes spurious machine-precision
entries from assembled operators, e.g. discrete gradients on cells
with non-affine geometry maps.
"""

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx.common import IndexMap
from dolfinx.cpp.la import SparsityPattern
from dolfinx.la import matrix_csr


def create_test_matrix(dtype):
    """Create a 4x4 (per process) matrix with nonzeros in a 2x2 block
    at (0..1, 0..1) and a 2x2 block at (2..3, 2..3).
    """
    im = IndexMap(MPI.COMM_WORLD, 4)
    sp = SparsityPattern(MPI.COMM_WORLD, [im, im], [1, 1])
    block = np.array([0, 1], dtype=np.int32)
    sp.insert(block, block)
    sp.insert(block + 2, block + 2)
    sp.finalize()
    return matrix_csr(sp, dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_insert_clamped(dtype):
    """Insert blocks of values containing entries that are negligible
    relative to the largest entry of the block and check that exactly
    those are stored as exact zeros, while all other entries pass
    through unchanged.
    """
    eps = np.finfo(dtype).eps  # Machine epsilon of the real type
    A = create_test_matrix(dtype)

    # For complex types insert rotated values so that clamping is
    # exercised on the complex magnitude
    phase = np.exp(0.7j).astype(dtype) if np.issubdtype(dtype, np.complexfloating) else dtype(1)

    rows01 = np.array([0, 1], dtype=np.int32)
    cols23 = np.array([2, 3], dtype=np.int32)

    # Set a 2x2 block in which one entry is negligible relative to the

    block0 = np.array([1.0, 10 * eps, 1e4 * eps, 0.5], dtype=dtype) * phase
    A.set_clamped(block0, rows01, rows01)

    block1 = np.array([1e-20, 1e-20 * 10 * eps], dtype=dtype) * phase
    A.set_clamped(block1, np.array([2], dtype=np.int32), cols23)

    block2 = np.array([100 * eps, 1.0], dtype=dtype) * phase
    A.add_clamped(block2, np.array([3], dtype=np.int32), cols23)
    A.add_clamped(block2, np.array([3], dtype=np.int32), cols23, C=10.0)

    expected = np.zeros((4, 4), dtype=dtype)
    expected[0, 0] = block0[0]
    expected[0, 1] = 0  # clamped: 10 * eps < 1000 * eps
    expected[1, 0] = block0[2]
    expected[1, 1] = block0[3]
    expected[2, 2] = block1[0]
    expected[2, 3] = 0  # clamped: negligible relative to block max
    expected[3, 2] = block2[0]  # clamped on first add, kept with C=10
    expected[3, 3] = block2[1] + block2[1]

    assert np.array_equal(A.to_dense(), expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_insert_clamped_all_zero(dtype):
    """Inserting an all-zero block must be well defined (the clamping
    tolerance is zero) and leave the block zero.
    """
    A = create_test_matrix(dtype)
    rows01 = np.array([0, 1], dtype=np.int32)
    block = np.zeros(4, dtype=dtype)
    A.set_clamped(block, rows01, rows01)
    A.add_clamped(block, rows01, rows01)
    assert np.array_equal(A.to_dense(), np.zeros((4, 4), dtype=dtype))
