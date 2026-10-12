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


@pytest.mark.parametrize(
    "factory_name",
    ["partitioner", "partitioner_scotch", "partitioner_parmetis", "partitioner_kahip"],
)
def test_partition_callable(factory_name) -> None:
    """Configured partitioners accept and return Python adjacency lists."""
    factory = getattr(graph, factory_name, None)
    if factory is None:
        pytest.skip(f"DOLFINx built without {factory_name}")
    part = factory()
    assert isinstance(part, graph.GraphPartitioner)
    g = dual_graph(MPI.COMM_SELF, 12)
    destinations = part(
        MPI.COMM_SELF,
        4,
        g,
        node_weights=np.ones(g.num_nodes, dtype=np.int32),
        edge_weights=np.ones(g.array.size, dtype=np.int32),
        ghosting=True,
    )
    assert isinstance(destinations, graph.AdjacencyList)
    assert destinations.num_nodes == g.num_nodes
    parts = owners(destinations)
    assert parts.min() >= 0 and parts.max() < 4
    for i in range(g.num_nodes):
        assert set(destinations.links(i)) == {parts[i]} | set(parts[g.links(i)])


def test_partition_callback() -> None:
    """Mesh callbacks receive Python graphs and can call a built-in partitioner."""
    part = graph.partitioner()
    called = False

    def callback(comm, nparts, local_graph, node_weights, edge_weights, ghosting):
        nonlocal called
        called = True
        assert isinstance(local_graph, graph.AdjacencyList)
        return part(comm, nparts, local_graph, node_weights, edge_weights, ghosting)

    msh = mesh.create_unit_square(MPI.COMM_WORLD, 12, 12, partitioner=callback)
    assert MPI.COMM_WORLD.allreduce(called, MPI.LOR)
    assert msh.topology.index_map(2).size_global == 2 * 12 * 12


def test_hybrid_callback() -> None:
    """Hybrid callbacks use Python graphs in direct calls and mesh creation."""
    part = graph.partitioner()
    called = False

    def callback(comm, nparts, local_graph, x, node_weights, edge_weights, ghosting):
        nonlocal called
        called = True
        assert isinstance(local_graph, graph.AdjacencyList)
        assert x.shape[0] == local_graph.num_nodes
        return part(comm, nparts, local_graph, node_weights, edge_weights, ghosting)

    hybrid = mesh.create_hybrid_cell_partitioner(callback)
    assert isinstance(hybrid, graph.HybridPartitioner)
    g = dual_graph(MPI.COMM_SELF, 12)
    destinations = hybrid(MPI.COMM_SELF, 4, g, np.zeros((g.num_nodes, 2)))
    assert called
    assert isinstance(destinations, graph.AdjacencyList)
    assert destinations.num_nodes == g.num_nodes

    called = False
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 12, 12, partitioner=hybrid)
    assert MPI.COMM_WORLD.allreduce(called, MPI.LOR)
    assert msh.topology.index_map(2).size_global == 2 * 12 * 12


@pytest.mark.parametrize("name", ["partition_morton", "partition_hilbert"])
def test_geometric_partitioner(name) -> None:
    """Geometric handles stay usable directly and through mesh creation."""
    part = getattr(graph, name)
    assert isinstance(part, graph.GeometricPartitioner)
    x = np.column_stack([np.linspace(0, 1, 32), np.zeros(32)])
    destinations = part(MPI.COMM_SELF, 4, x)
    assert destinations.shape == (32,)
    assert np.array_equal(np.bincount(destinations), np.full(4, 8))
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 12, 12, partitioner=part)
    assert msh.topology.index_map(2).size_global == 2 * 12 * 12


def test_geometric_callback() -> None:
    """A custom geometric handle is callable and accepted by mesh creation."""
    called = False

    def callback(comm, nparts, x, node_weights):
        nonlocal called
        called = True
        return graph.partition_morton(comm, nparts, x, node_weights)

    part = mesh.create_geometric_cell_partitioner(callback)
    assert isinstance(part, graph.GeometricPartitioner)
    destinations = part(MPI.COMM_SELF, 4, np.zeros((32, 2)))
    assert called
    assert destinations.shape == (32,)
    called = False
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 12, 12, partitioner=part)
    assert called
    assert msh.topology.index_map(2).size_global == 2 * 12 * 12


@pytest.mark.skipif(
    not hasattr(graph, "partitioner_parmetis_hybrid"), reason="DOLFINx built without ParMETIS"
)
def test_builtin_hybrid_partitioner() -> None:
    """The configured hybrid factory returns a Python handle."""
    part = graph.partitioner_parmetis_hybrid(imbalance=1.05, options=(1, 0, 5))
    assert isinstance(part, graph.HybridPartitioner)
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 12, 12, partitioner=part)
    assert msh.topology.index_map(2).size_global == 2 * 12 * 12
