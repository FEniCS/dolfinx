# Copyright (C) 20222 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for MatrixCSR"""
import dolfinx.common
from mpi4py import MPI
import pytest

import numpy as np

from dolfinx import cpp as _cpp
from dolfinx import la
from dolfinx.fem import functionspace
from dolfinx.mesh import create_unit_square


def test_create_matrix_csr():
    """Test creation of CSR matrix with specified types"""
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 11)
    V = functionspace(mesh, ("Lagrange", 1))
    map = V.dofmap.index_map
    bs = V.dofmap.index_map_bs

    pattern = _cpp.la.SparsityPattern(mesh.comm, [map, map], [bs, bs])
    rows = np.arange(0, bs * map.size_local)
    cols = np.arange(0, bs * map.size_local)
    pattern.insert(rows, cols)
    pattern.finalize()

    A = la.matrix_csr(pattern)
    assert A.data.dtype == np.float64
    A = la.matrix_csr(pattern, dtype=np.float64)
    assert A.data.dtype == np.float64

    A = la.matrix_csr(pattern, dtype=np.float32)
    assert A.data.dtype == np.float32

    A = la.matrix_csr(pattern, dtype=np.complex128)
    assert A.data.dtype == np.complex128

    cmap = pattern.column_index_map()
    num_cols = cmap.size_local + cmap.num_ghosts
    num_rows = bs * (map.size_local + map.num_ghosts)
    zero = np.zeros((num_rows, bs * num_cols), dtype=np.complex128)
    assert np.allclose(A.to_dense(), zero)


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
        np.int8,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
    ],
)
def test_create_vector(dtype):
    """Test creation of a distributed vector"""
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)
    im = mesh.topology.index_map(0)

    for bs in range(1, 4):
        x = la.vector(im, bs=bs, dtype=dtype)
        assert x.array.dtype == dtype
        assert x.array.size == bs * (im.size_local + im.num_ghosts)
