# Copyright (C) 2026 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for graph partitioning."""

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx import graph, mesh


def dual_graph(comm, n):
    """Dual graph of a unit square mesh of n x n squares."""
    msh = mesh.create_unit_square(comm, n, n)
    tdim = msh.topology.dim
    num_cells = msh.topology.index_map(tdim).size_local
    c_to_v = msh.topology.connectivity(tdim, 0)
    vertices = msh.topology.index_map(0).local_to_global(c_to_v.array)
    cells = graph.adjacencylist(vertices.reshape(c_to_v.num_nodes, -1)[:num_cells])
    return mesh.build_dual_graph(comm, msh.topology.cell_type, cells)


def owners(destinations):
    """Owning part of each node."""
    return np.array([destinations.links(i)[0] for i in range(destinations.num_nodes)])


@pytest.mark.parametrize("nparts", [2, 7])
def test_partition_serial(nparts) -> None:
    """A graph held on one process is split into nparts parts."""
    g = dual_graph(MPI.COMM_SELF, 12)
    destinations = graph.partition(MPI.COMM_SELF, nparts, g)
    assert destinations.num_nodes == g.num_nodes

    parts = owners(destinations)
    assert parts.min() >= 0 and parts.max() < nparts
    counts = np.bincount(parts, minlength=nparts)
    assert counts.min() > 0, "a part is empty"
    assert counts.max() <= 2 * counts.min(), "parts are poorly balanced"


def test_partition_distributed() -> None:
    """A distributed graph is partitioned across the communicator."""
    comm = MPI.COMM_WORLD
    g = dual_graph(comm, 12)
    destinations = graph.partition(comm, comm.size, g)
    assert destinations.num_nodes == g.num_nodes

    parts = owners(destinations)
    assert parts.min() >= 0 and parts.max() < comm.size
    assert comm.allreduce(destinations.num_nodes, MPI.SUM) == 2 * 12 * 12


def test_partition_ghosting() -> None:
    """Ghosting reports the parts sharing a node, the owner first."""
    g = dual_graph(MPI.COMM_SELF, 12)
    ghosted = graph.partition(MPI.COMM_SELF, 4, g, ghosting=True)
    assert ghosted.num_nodes == g.num_nodes
    parts = owners(ghosted)
    assert parts.min() >= 0 and parts.max() < 4
    for i in range(g.num_nodes):
        expected = {parts[i]} | set(parts[g.links(i)])
        destinations = ghosted.links(i)
        assert set(destinations) == expected
        assert len(destinations) == len(expected)


@pytest.mark.skipif(not hasattr(graph, "partitioner_scotch"), reason="DOLFINx built without SCOTCH")
def test_partition_explicit_partitioner() -> None:
    """A partitioner can be chosen explicitly."""
    g = dual_graph(MPI.COMM_SELF, 12)
    destinations = graph.partition(MPI.COMM_SELF, 4, g, partitioner=graph.partitioner_scotch())
    parts = owners(destinations)
    assert np.bincount(parts, minlength=4).min() > 0
