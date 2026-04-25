# Copyright (C) 2026 Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Shared pytest fixtures for dolfinx.la unit tests."""

from mpi4py import MPI

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import dolfinx
import dolfinx.cpp as _cpp
from dolfinx.mesh import create_unit_square


@pytest.fixture
def mat_gather():
    """Return a function that gathers a distributed MatrixCSR onto all
    processes as a scipy CSR matrix (in global column indexing).
    """

    def _mat_gather(A):
        nr = A.index_map(0).size_local
        gatheredvals = np.concatenate(MPI.COMM_WORLD.allgather(A.data[: A.indptr[nr]]))
        gatheredptrs = MPI.COMM_WORLD.allgather(A.indptr[: nr + 1])
        cols = A.index_map(1).local_to_global(A.indices[: A.indptr[nr]])
        gatheredcols = np.concatenate(MPI.COMM_WORLD.allgather(cols))
        indptr = gatheredptrs[0]
        for i in range(1, len(gatheredptrs)):
            indptr = np.concatenate((indptr, (gatheredptrs[i][1:] + indptr[-1])))
        return csr_matrix((gatheredvals, gatheredcols, indptr))

    return _mat_gather


@pytest.fixture
def mat_random():
    """Return a function that creates a random (dense) MatrixCSR for testing."""

    def _mat_random(dim0, dim1, seed, dtype):
        mesh = create_unit_square(MPI.COMM_WORLD, 5, 4)
        mesh.topology.create_entities(1)
        imap0 = mesh.topology.index_map(dim0)
        imap1 = mesh.topology.index_map(dim1)
        sp = _cpp.la.SparsityPattern(mesh.comm, [imap0, imap1], [1, 1])
        rows = np.arange(0, imap0.size_local)
        cols = np.arange(0, imap1.size_local + imap1.num_ghosts)
        sp.insert(rows, cols)
        sp.finalize()

        A = dolfinx.la.matrix_csr(sp, dtype=dtype)
        rng = np.random.default_rng(seed)
        A.data[:] = rng.random(len(A.data))
        return A

    return _mat_random
