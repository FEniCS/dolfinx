# Copyright (C) 2026 Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for MatrixCSR."""

import numpy as np
import pytest


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ],
)
def test_matmul(dtype, mat_random, mat_gather):
    # Create two random square MatrixCSR
    A = mat_random(0, 0, 12345, dtype)
    B = mat_random(0, 0, 54321, dtype)

    Ascipy = mat_gather(A)
    Bscipy = mat_gather(B)
    lr0, lr1 = A.index_map(0).local_range
    nr = A.index_map(0).size_local
    # Check gathered matrix
    assert np.allclose(A.to_dense()[:nr, :], Ascipy.todense()[lr0:lr1])
    lrB0, lrB1 = B.index_map(0).local_range
    nrB = B.index_map(0).size_local
    # Check gathered matrix
    assert np.allclose(B.to_dense()[:nrB, :], Bscipy.todense()[lrB0:lrB1])

    C = A.matmul(B)
    lrC0, lrC1 = C.index_map(0).local_range
    nrC = C.index_map(0).size_local
    Cscipy = Ascipy @ Bscipy
    assert np.allclose(C.to_dense()[:nrC, :], Cscipy.todense()[lrC0:lrC1])


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ],
)
def test_matmul_rect(dtype, mat_random, mat_gather):
    # Create two random rectangular MatrixCSR
    A = mat_random(0, 1, 12345, dtype)
    B = mat_random(1, 0, 54321, dtype)

    Ascipy = mat_gather(A)
    Bscipy = mat_gather(B)
    lr0, lr1 = A.index_map(0).local_range
    nr = A.index_map(0).size_local
    # Check gathered matrix
    assert np.allclose(A.to_dense()[:nr, :], Ascipy.todense()[lr0:lr1])
    lrB0, lrB1 = B.index_map(0).local_range
    nrB = B.index_map(0).size_local
    # Check gathered matrix
    assert np.allclose(B.to_dense()[:nrB, :], Bscipy.todense()[lrB0:lrB1])

    C = A.matmul(B)
    lrC0, lrC1 = C.index_map(0).local_range
    nrC = C.index_map(0).size_local
    Cscipy = Ascipy @ Bscipy
    assert np.allclose(C.to_dense()[:nrC, :], Cscipy.todense()[lrC0:lrC1])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_matmul_zeros(dtype, mat_random, mat_gather):
    A = mat_random(0, 0, 123, dtype)
    B = mat_random(0, 0, 321, dtype)

    # Create structural zeros in B
    pos = 0
    for i in range(B.index_map(0).size_local):
        for j in range(B.indptr[i], B.indptr[i + 1]):
            if i != B.indices[j]:
                B.data[pos] = 0.0
            pos += 1
    As = mat_gather(A)
    Bs = mat_gather(B)

    Cs = As @ Bs
    C0 = A.matmul(B)

    C = mat_gather(C0)
    assert C.nnz == Cs.nnz
    assert np.allclose(Cs.todense(), C.todense())


def test_bad_shape(mat_random):
    # Test matmul of incompatible matrices (should raise an error)
    A = mat_random(0, 1, 12345, np.float64)
    B = mat_random(0, 2, 54321, np.float64)

    with pytest.raises(RuntimeError):
        A.matmul(B)

    A = mat_random(0, 2, 12345, np.float64)
    B = mat_random(1, 0, 54321, np.float64)

    with pytest.raises(RuntimeError):
        A.matmul(B)
