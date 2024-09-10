# Copyright (C) 2024 Paul Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx import mesh


@pytest.mark.parametrize("n", [2, 10, 100])
@pytest.mark.parametrize(
    "ghost_mode", [mesh.GhostMode.none, mesh.GhostMode.shared_vertex, mesh.GhostMode.shared_facet]
)
@pytest.mark.parametrize(
    "ghost_mode_refined",
    [mesh.GhostMode.none, mesh.GhostMode.shared_vertex, mesh.GhostMode.shared_facet],
)
@pytest.mark.parametrize("redistribute", [True, False])
def test_refine_interval(n, ghost_mode, redistribute, ghost_mode_refined):
    msh = mesh.create_interval(MPI.COMM_WORLD, n, [0, 1], ghost_mode=ghost_mode)
    msh_refined, edges = mesh.refine_interval(
        msh, redistribute=redistribute, ghost_mode=ghost_mode_refined
    )

    # vertex count
    assert msh_refined.topology.index_map(0).size_global == 2 * n + 1

    # edge count
    edge_count = np.array([len(edges)], dtype=np.int64)
    edge_count = MPI.COMM_WORLD.allreduce(edge_count)

    assert edge_count == msh_refined.topology.index_map(1).size_global == 2 * n


@pytest.mark.parametrize("n", [50, 100])
@pytest.mark.parametrize(
    "ghost_mode", [mesh.GhostMode.none, mesh.GhostMode.shared_vertex, mesh.GhostMode.shared_facet]
)
@pytest.mark.parametrize(
    "ghost_mode_refined",
    [mesh.GhostMode.none, mesh.GhostMode.shared_vertex, mesh.GhostMode.shared_facet],
)
@pytest.mark.parametrize("redistribute", [True, False])
def test_refine_interval_adaptive(n, ghost_mode, redistribute, ghost_mode_refined):
    msh = mesh.create_interval(MPI.COMM_WORLD, n, [0, 1], ghost_mode=ghost_mode)
    msh_refined, edges = mesh.refine_interval(
        msh, np.arange(10, dtype=np.int32), redistribute=redistribute, ghost_mode=ghost_mode_refined
    )

    # vertex count
    assert msh_refined.topology.index_map(0).size_global == n + 1 + 10 * MPI.COMM_WORLD.size

    # edge count
    edge_count = np.array([len(edges)], dtype=np.int64)
    edge_count = MPI.COMM_WORLD.allreduce(edge_count)

    assert (
        edge_count == msh_refined.topology.index_map(1).size_global == n + 10 * MPI.COMM_WORLD.size
    )
