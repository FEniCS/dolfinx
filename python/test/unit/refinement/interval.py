# Copyright (C) 2024 Paul Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx import mesh


@pytest.mark.parametrize(
    "ghost_mode", [mesh.GhostMode.none, mesh.GhostMode.shared_vertex, mesh.GhostMode.shared_facet]
)
@pytest.mark.parametrize("redistribute", [True, False])
def test_refine_interval(ghost_mode, redistribute):
    msh = mesh.create_interval(MPI.COMM_WORLD, 100, [0, 1], ghost_mode=ghost_mode)
    msh_refined, edges = mesh.refine_interval(msh, redistribute=redistribute)

    # vertex count
    assert msh_refined.topology.index_map(0).size_global == 201

    # edge count
    edge_count = np.array([len(edges)], dtype=np.int64)
    edge_count = MPI.COMM_WORLD.allreduce(edge_count)

    assert edge_count == msh_refined.topology.index_map(1).size_global == 200


@pytest.mark.parametrize(
    "ghost_mode", [mesh.GhostMode.none, mesh.GhostMode.shared_vertex, mesh.GhostMode.shared_facet]
)
@pytest.mark.parametrize("redistribute", [True, False])
def test_refine_interval_adaptive(ghost_mode, redistribute):
    msh = mesh.create_interval(MPI.COMM_WORLD, 100, [0, 1], ghost_mode=ghost_mode)
    msh_refined, edges = mesh.refine_interval(msh, np.arange(10), redistribute=redistribute)

    # vertex count
    assert msh_refined.topology.index_map(0).size_global == 101 + 10 * MPI.COMM_WORLD.size

    # edge count
    edge_count = np.array([len(edges)], dtype=np.int64)
    edge_count = MPI.COMM_WORLD.allreduce(edge_count)

    assert (
        edge_count
        == msh_refined.topology.index_map(1).size_global
        == 100 + 10 * MPI.COMM_WORLD.size
    )
