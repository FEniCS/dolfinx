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
