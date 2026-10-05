# Copyright (C) 2026 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for PETSc matrix creation from a sparsity pattern."""

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx.common import index_map as create_index_map
from dolfinx.la import sparsity_pattern


@pytest.mark.petsc4py
@pytest.mark.parametrize("mat_type", ["aij", "is"])
def test_shared_index_map_shares_lgmap(mat_type):
    """One index map for both dimensions gives one local-to-global map.

    ``la::petsc::create_matrix`` attaches the row mapping to both
    dimensions when the pattern's row and column maps are the same
    object. Column ghosts added by finalization must not split them.
    """
    from dolfinx.cpp.la.petsc import create_matrix  # noqa: TID251

    comm = MPI.COMM_WORLD
    n = 4
    ghosts = (
        np.array([0] if comm.rank > 0 else [], dtype=np.int64),
        np.array([0] if comm.rank > 0 else [], dtype=np.int32),
    )
    imap = create_index_map(comm, n, ghosts)
    pattern = sparsity_pattern(comm, [imap, imap], [1, 1])
    pattern.insert_diagonal(np.arange(n, dtype=np.int32))

    # Add to rank 0's row at a column rank 0 does not hold, which makes
    # finalization grow the column map
    if comm.rank > 0:
        pattern.insert(n, 0)
    pattern.finalize()

    # Each other rank adds one ghost column to rank 0. Reduce before
    # asserting, so that a failure cannot leave ranks out of step.
    grown = pattern.index_map(1).num_ghosts - pattern.input_index_map(1).num_ghosts
    A = create_matrix(comm, pattern._cpp_object, mat_type)
    assert comm.allreduce(grown, MPI.SUM) == comm.size - 1
    assert pattern.input_index_map(1).num_ghosts == imap.num_ghosts
    rmap, cmap = A.getLGMap()
    assert rmap.handle == cmap.handle
    A.destroy()
