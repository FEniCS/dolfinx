# Copyright (C) 2022 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for sparsity pattern creation."""

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx.common import index_map as create_index_map
from dolfinx.fem import functionspace, locate_dofs_topological
from dolfinx.la import sparsity_pattern, sparsity_pattern_blocked
from dolfinx.mesh import create_unit_square, exterior_facet_indices


def test_add_diagonal():
    """Test adding entries to diagonal of sparsity pattern."""
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    gdim = mesh.geometry.dim
    V = functionspace(mesh, ("Lagrange", 1, (gdim,)))
    pattern = sparsity_pattern(
        mesh.comm,
        [V.dofmap.index_map, V.dofmap.index_map],
        [V.dofmap.index_map_bs, V.dofmap.index_map_bs],
    )
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    facets = exterior_facet_indices(mesh.topology)
    blocks = locate_dofs_topological(V, mesh.topology.dim - 1, facets)
    pattern.insert_diagonal(blocks)
    pattern.finalize()
    assert len(blocks) == pattern.num_nonzeros


def test_blocked_pattern_with_empty_blocks():
    """Test creation of a blocked pattern with structural zero blocks."""
    # COMM_SELF: the block structure under test is process-local and
    # involves no cross-rank communication, so the test runs unmodified
    # under any number of MPI ranks.
    index_map = create_index_map(MPI.COMM_SELF, 2)
    pattern = sparsity_pattern(MPI.COMM_SELF, [index_map, index_map], [1, 1])
    blocked_pattern = sparsity_pattern_blocked(
        MPI.COMM_SELF,
        [[pattern, None], [None, pattern]],
        [[(index_map, 1), (index_map, 1)], [(index_map, 1), (index_map, 1)]],
        [[1, 1], [1, 1]],
    )
    blocked_pattern.finalize()
    assert blocked_pattern.num_nonzeros == 0


def test_column_index_map_growth():
    """Finalizing can add column ghosts without changing the input maps.

    Rank 1 assembles an entry on a row it does not own, at a column the
    row owner does not hold, so finalization adds a column ghost on
    rank 0. ``index_map`` must still return the constructor's maps, so
    that a pattern built from a single map is recognisable as such.
    """
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("Requires at least two MPI ranks")

    ghosts = (
        np.array([0] if comm.rank == 1 else [], dtype=np.int64),
        np.array([0] if comm.rank == 1 else [], dtype=np.int32),
    )
    imap = create_index_map(comm, 1, ghosts)
    pattern = sparsity_pattern(comm, [imap, imap], [1, 1])

    # Rank 1 adds to rank 0's row, at a column rank 0 does not hold
    if comm.rank == 1:
        pattern.insert(1, 0)
    pattern.finalize()

    for dim in range(2):
        assert pattern.index_map(dim).num_ghosts == imap.num_ghosts
    assert pattern.column_index_map().size_local == imap.size_local
    if comm.rank == 0:
        assert pattern.column_index_map().num_ghosts == 1
        assert pattern.column_index_map().ghosts[0] == 1
