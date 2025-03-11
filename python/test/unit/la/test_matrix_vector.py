# Copyright (C) 20222 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for MatrixCSR"""

from mpi4py import MPI

import numpy as np
import pytest

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

    dense = A.to_dense()
    assert np.allclose(dense, np.zeros(dense.shape, dtype=np.complex128))


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ],
)
def test_matvec(dtype):
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 3)
    imap = mesh.topology.index_map(0)
    sp = _cpp.la.SparsityPattern(mesh.comm, [imap, imap], [1, 1])
    rows = np.arange(0, imap.size_local)
    cols = np.arange(0, imap.size_local)
    sp.insert(rows, cols)
    sp.finalize()

    # Identity
    A = la.matrix_csr(sp, dtype=dtype)
    for i in range(imap.size_local):
        A.add([2.0], [i], [i])
    A.scatter_reverse()

    b = la.vector(imap, dtype=dtype)
    u = la.vector(imap, dtype=dtype)
    b.array[:] = 1.0
    A.mult(b, u)
    u.scatter_forward()
    assert np.allclose(u.array[: imap.size_local], 2.0)


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
        np.int8,
        np.int32,
        np.int64,
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


def xfail_norm_of_integral_type_vector(dtype):
    return pytest.param(
        dtype,
        marks=pytest.mark.xfail(
            reason="Norm of vector of integers not implemented", strict=True, raises=TypeError
        ),
    )


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
        xfail_norm_of_integral_type_vector(np.int8),
        xfail_norm_of_integral_type_vector(np.int32),
        xfail_norm_of_integral_type_vector(np.int64),
    ],
)
@pytest.mark.parametrize(
    "norm_type",
    [
        la.Norm.l1,
        la.Norm.l2,
        la.Norm.linf,
        pytest.param(
            la.Norm.frobenius,
            marks=pytest.mark.xfail(reason="Norm type not supported for vector", strict=True),
        ),
    ],
)
def test_vector_norm(dtype, norm_type):
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)
    im = mesh.topology.index_map(0)
    x = la.vector(im, dtype=dtype)
    x.array[:] = 0.0
    normed_value = la.norm(x, norm_type)
    assert np.isclose(normed_value, 0.0)
