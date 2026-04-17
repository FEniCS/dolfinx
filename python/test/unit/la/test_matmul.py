# Copyright (C) 2026 Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for MatrixCSR."""

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx import cpp as _cpp
from dolfinx import la
from dolfinx.mesh import create_unit_square


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ],
)
def test_matmul(dtype, mat_gather):
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 3)
    imap = mesh.topology.index_map(0)
    sp = _cpp.la.SparsityPattern(mesh.comm, [imap, imap], [1, 1])
    rows = np.arange(0, imap.size_local)
    cols = np.arange(0, imap.size_local + imap.num_ghosts)
    sp.insert(rows, cols)
    sp.finalize()

    # Identity
    A = la.matrix_csr(sp, dtype=dtype)
    B = la.matrix_csr(sp, dtype=dtype)
    rng = np.random.default_rng(12345)
    A.data[:] = rng.random(len(A.data))
    A.scatter_reverse()
    B.data[:] = rng.random(len(B.data))
    B.scatter_reverse()

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
