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
def test_transpose_square(dtype, mat_random, mat_gather):
    # Create random square MatrixCSR
    A = mat_random(0, 0, 12345, dtype)

    Ascipy = mat_gather(A)
    lr0, lr1 = A.index_map(0).local_range
    nr = A.index_map(0).size_local
    # Check gathered matrix
    assert np.allclose(A.to_dense()[:nr, :], Ascipy.todense()[lr0:lr1])

    AT = A.transpose()
    AscipyT = Ascipy.T
    lr0, lr1 = AT.index_map(0).local_range
    nr = AT.index_map(0).size_local

    assert np.allclose(AT.to_dense()[:nr, :], AscipyT.todense()[lr0:lr1])


@pytest.mark.parametrize("bs", [[1, 1], [2, 2], [3, 3], [2, 3], [3, 2]])
@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ],
)
def test_transpose_block(dtype, bs, mat_random, mat_gather):
    # Create random rectangular MatrixCSR with various block sizes
    A = mat_random(0, 1, 12345, dtype, bs)
    Ascipy = mat_gather(A)

    lr0, lr1 = A.index_map(0).local_range
    nr = A.index_map(0).size_local

    # Check gathered matrix
    assert np.allclose(A.to_dense()[: nr * bs[0], :], Ascipy.todense()[lr0 * bs[0] : lr1 * bs[0]])

    AT = A.transpose()
    AscipyT = Ascipy.T

    lr0, lr1 = AT.index_map(0).local_range
    nr = AT.index_map(0).size_local

    assert np.allclose(AT.to_dense()[: nr * bs[1], :], AscipyT.todense()[lr0 * bs[1] : lr1 * bs[1]])
