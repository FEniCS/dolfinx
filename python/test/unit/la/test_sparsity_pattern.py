# Copyright (C) 2022 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for sparsity pattern creation."""

from mpi4py import MPI

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
