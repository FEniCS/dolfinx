# Copyright (C) 2021-2026 Garth N. Wells and Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Graph representations and operations on graphs."""

from collections.abc import Callable
from functools import wraps
from typing import Generic, ParamSpec, TypeVar

from mpi4py import MPI as _MPI

import numpy as np
import numpy.typing as npt

from dolfinx import cpp as _cpp
from dolfinx.common import IndexMap
from dolfinx.cpp.graph import (
    reorder_hilbert,
    reorder_morton,
)
from dolfinx.typing import Index

__all__ = [
    "AdjacencyList",
    "GeometricPartitioner",
    "GraphPartitioner",
    "HybridPartitioner",
    "adjacencylist",
    "comm_graph",
    "comm_graph_data",
    "comm_to_json",
    "distribute",
    "partition",
    "partition_hilbert",
    "partition_morton",
    "partitioner",
    "reorder_hilbert",
    "reorder_morton",
    "reorder_rcm",
]


class AdjacencyList(Generic[Index]):
    """Adjacency list representation of a graph."""

    _cpp_object: (
        _cpp.graph.AdjacencyList_int32
        | _cpp.graph.AdjacencyList_int64
        | _cpp.graph.AdjacencyList_int_sizet_int8__int32_int32
    )

    def __init__(
        self,
        g: (
            _cpp.graph.AdjacencyList_int32
            | _cpp.graph.AdjacencyList_int64
            | _cpp.graph.AdjacencyList_int_sizet_int8__int32_int32
        ),
    ):
        """Creates a Python wrapper for the exported adjacency list class.

        Note:
            Do not use this constructor directly. Instead use
            :func:`adjacencylist`.

        Args:
            g: The underlying cpp instance that this object will wrap.
        """
        self._cpp_object = g

    def __eq__(self, other: object) -> bool:
        """Check that two wrappers hold the same adjacency list."""
        if not isinstance(other, AdjacencyList):
            return NotImplemented
        return self._cpp_object == other._cpp_object

    def __repr__(self) -> str:
        """String representation of the adjacency list."""
        return self._cpp_object.__repr__()

    def links(self, node: int) -> npt.NDArray[Index]:
        """Retrieve the links of a node.

        Note:
            This is available only for adjacency lists with no
            additional link (edge) data.

        Args:
            node: Node to retrieve the connectivity of.

        Returns:
            Neighbors of the node.
        """
        return self._cpp_object.links(node)  # type: ignore[union-attr,return-value]

    @property
    def array(self) -> npt.NDArray[Index]:
        """Array representation of the adjacency list.

        Note:
            This is available only for adjacency lists with no
            additional link (edge) data.

        Returns:
            Flattened array representation of the adjacency list.
        """
        return self._cpp_object.array  # type: ignore[union-attr,return-value]

    @property
    def offsets(self) -> npt.NDArray[np.int32]:
        """Offsets for each node in the :func:`array`.

        Returns:
            Array of indices with shape `(num_nodes+1)`.
        """
        return self._cpp_object.offsets

    @property
    def num_nodes(self) -> np.int32:
        """Number of nodes in the adjacency list.

        Returns:
            Number of nodes.
        """
        return self._cpp_object.num_nodes  # type: ignore[return-value]


def adjacencylist(
    data: npt.NDArray[Index], offsets: npt.NDArray[np.int32] | None = None
) -> AdjacencyList[Index]:
    """Create an :class:`AdjacencyList` for `int32` or `int64` datasets.

    Args:
        data: The adjacency array. If the array is one-dimensional,
            offsets should be supplied. If the array is two-dimensional
            the number of edges per node is the second dimension.
        offsets: The offsets array with the number of edges per node.

    Returns:
        An adjacency list.
    """
    # TODO: Switch to np.isdtype(data.dtype, np.int32) once numpy >= 2.0 is
    # enforced
    cpp_t: type[_cpp.graph.AdjacencyList_int32] | type[_cpp.graph.AdjacencyList_int64]
    if data.dtype == np.int32:
        cpp_t = _cpp.graph.AdjacencyList_int32
    elif data.dtype == np.int64:
        cpp_t = _cpp.graph.AdjacencyList_int64
    else:
        raise TypeError("Data type for adjacency list not supported.")

    cpp_object = cpp_t(data, offsets) if offsets is not None else cpp_t(data)  # type: ignore[arg-type]
    return AdjacencyList(cpp_object)


class GraphPartitioner:
    """A graph partitioner for direct calls and mesh creation."""

    def __init__(self, partitioner: _cpp.graph.GraphPartitioner):
        """Wrap a native handle. Use the partitioner factories instead."""
        self._cpp_object = partitioner

    def __call__(
        self,
        comm: _MPI.Comm,
        nparts: int,
        local_graph: AdjacencyList[np.int64],
        node_weights: npt.NDArray[np.int32] | None = None,
        edge_weights: npt.NDArray[np.int32] | None = None,
        ghosting: bool = False,
    ) -> AdjacencyList[np.int32]:
        """Partition a distributed graph.

        Args:
            comm: Communicator over which the graph is distributed.
            nparts: Number of parts.
            local_graph: Owned nodes with global node indices as links.
            node_weights: Node weights; None gives equal weights.
            edge_weights: One weight per link, or equal weights if None.
            ghosting: Include parts owning adjacent nodes as destinations.

        Returns:
            Destinations for each node, with its owning part first.
        """
        cpp_graph = local_graph._cpp_object
        if not isinstance(cpp_graph, _cpp.graph.AdjacencyList_int64):
            raise TypeError("Partitioning requires an adjacency list with int64 links.")
        return AdjacencyList(
            self._cpp_object(comm, nparts, cpp_graph, node_weights, edge_weights, ghosting)
        )


class GeometricPartitioner:
    """A partitioner using positions, without graph edges or ghosting."""

    def __init__(self, partitioner: _cpp.graph.GeometricPartitioner):
        """Wrap a native handle. Use the partitioner factories instead."""
        self._cpp_object = partitioner

    def __call__(
        self,
        comm: _MPI.Comm,
        nparts: int,
        x: npt.NDArray[np.float64],
        node_weights: npt.NDArray[np.int32] | None = None,
    ) -> npt.NDArray[np.int32]:
        """Partition nodes by their positions.

        Args:
            comm: Communicator over which the nodes are distributed.
            nparts: Number of parts.
            x: Positions, with one row per owned node.
            node_weights: Node weights; None gives equal weights.

        Returns:
            Owning part of each node.
        """
        return self._cpp_object(comm, nparts, x, node_weights)


class HybridPartitioner:
    """A partitioner using both graph edges and node positions."""

    def __init__(self, partitioner: _cpp.graph.HybridPartitioner):
        """Wrap a native handle. Use the partitioner factories instead."""
        self._cpp_object = partitioner

    def __call__(
        self,
        comm: _MPI.Comm,
        nparts: int,
        dual_graph: AdjacencyList[np.int64],
        x: npt.NDArray[np.float64],
        node_weights: npt.NDArray[np.int32] | None = None,
        edge_weights: npt.NDArray[np.int32] | None = None,
        ghosting: bool = False,
    ) -> AdjacencyList[np.int32]:
        """Partition a graph using its edges and node positions.

        Args:
            comm: Communicator over which the graph is distributed.
            nparts: Number of parts.
            dual_graph: Owned nodes with global node indices as links.
            x: Positions, with one row per owned node.
            node_weights: Node weights; None gives equal weights.
            edge_weights: One weight per link, or equal weights if None.
            ghosting: Include parts owning adjacent nodes as destinations.

        Returns:
            Destinations for each node, with its owning part first.
        """
        cpp_graph = dual_graph._cpp_object
        if not isinstance(cpp_graph, _cpp.graph.AdjacencyList_int64):
            raise TypeError("Partitioning requires an adjacency list with int64 links.")
        return AdjacencyList(
            self._cpp_object(comm, nparts, cpp_graph, x, node_weights, edge_weights, ghosting)
        )


def partitioner() -> GraphPartitioner:
    """Create the default graph partitioner, also used by mesh creation."""
    return GraphPartitioner(_cpp.graph.partitioner())


_P = ParamSpec("_P")
_C = TypeVar("_C")
_T = TypeVar("_T")


def _wrap_partitioner_factory(
    factory: Callable[_P, _C], wrapper: Callable[[_C], _T]
) -> Callable[_P, _T]:
    """Wrap native handles, preserving backend configuration signatures."""

    @wraps(factory)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        return wrapper(factory(*args, **kwargs))

    return wrapped


# Backend availability depends on the build, including its generated stubs.
_partitioner_scotch = getattr(_cpp.graph, "partitioner_scotch", None)
if _partitioner_scotch is not None:
    partitioner_scotch = _wrap_partitioner_factory(_partitioner_scotch, GraphPartitioner)
_partitioner_parmetis = getattr(_cpp.graph, "partitioner_parmetis", None)
if _partitioner_parmetis is not None:
    partitioner_parmetis = _wrap_partitioner_factory(_partitioner_parmetis, GraphPartitioner)
_partitioner_kahip = getattr(_cpp.graph, "partitioner_kahip", None)
if _partitioner_kahip is not None:
    partitioner_kahip = _wrap_partitioner_factory(_partitioner_kahip, GraphPartitioner)

partition_morton = GeometricPartitioner(_cpp.graph.partition_morton)
partition_hilbert = GeometricPartitioner(_cpp.graph.partition_hilbert)
_partitioner_parmetis_geom = getattr(_cpp.graph, "partitioner_parmetis_geom", None)
if _partitioner_parmetis_geom is not None:
    partitioner_parmetis_geom = GeometricPartitioner(_partitioner_parmetis_geom)
_partitioner_parmetis_hybrid = getattr(_cpp.graph, "partitioner_parmetis_hybrid", None)
if _partitioner_parmetis_hybrid is not None:
    partitioner_parmetis_hybrid = _wrap_partitioner_factory(
        _partitioner_parmetis_hybrid, HybridPartitioner
    )


def reorder_rcm(graph: AdjacencyList[np.int32]) -> npt.NDArray[np.int32]:
    """Re-order a graph using the reverse Cuthill-McKee algorithm.

    Pass it as :func:`create_mesh <dolfinx.mesh.create_mesh>`'s
    ``reorder_fn`` argument; it is also the default cell reordering.

    Args:
        graph: Graph to re-order.

    Returns:
        New index of each node, i.e. entry ``i`` is the new index of
        node ``i``.
    """
    return np.asarray(_cpp.graph.reorder_rcm(graph._cpp_object), dtype=np.int32)  # type: ignore[arg-type]


def partition(
    comm: _MPI.Comm,
    nparts: int,
    local_graph: AdjacencyList[np.int64],
    *,
    partitioner: GraphPartitioner | None = None,
    node_weights: npt.NDArray[np.int32] | None = None,
    edge_weights: npt.NDArray[np.int32] | None = None,
    ghosting: bool = False,
) -> AdjacencyList[np.int32]:
    """Partition the nodes of a graph into parts.

    Applies a partitioner to a graph, e.g. the dual graph of a mesh
    from :func:`build_dual_graph <dolfinx.mesh.build_dual_graph>`.
    :func:`create_mesh <dolfinx.mesh.create_mesh>` partitions a mesh
    itself; this is for partitioning a graph directly, for instance to
    build the subdomains of a domain decomposition method. Passing
    ``MPI.COMM_SELF`` partitions a graph held on one process into
    ``nparts`` parts without communication.

    Note:
        Collective across ``comm``, and requires DOLFINx to be built
        with a graph partitioner (ParMETIS, SCOTCH or KaHIP).

    Args:
        comm: MPI communicator that ``local_graph`` is distributed
            across.
        nparts: Number of parts to partition the graph into.
        local_graph: Graph to partition, with the nodes owned by this
            process as its nodes and global node indices as its links.
        partitioner: Partitioner to apply. The default partitioner,
            which is the one :func:`create_mesh
            <dolfinx.mesh.create_mesh>` uses, is applied if ``None``.
        node_weights: Weight of each node, which a partitioner may use
            to balance the parts.
        edge_weights: Weight of each edge, which a partitioner may use
            to weigh the cost of cutting it.
        ghosting: Return, for each node, the parts that share a node
            with it in addition to the part that owns it.

    Returns:
        Part(s) of each node of ``local_graph``. The first entry for a
        node is the part that owns it; any further entries are parts
        that ghost it, and appear only if ``ghosting`` is ``True``.
    """
    fn = GraphPartitioner(_cpp.graph.partitioner()) if partitioner is None else partitioner
    return fn(comm, nparts, local_graph, node_weights, edge_weights, ghosting)


def distribute(
    comm: _MPI.Comm,
    list: npt.NDArray[np.int64],
    destinations: AdjacencyList[np.int32],
) -> tuple[
    npt.NDArray[np.int64], npt.NDArray[np.int32], npt.NDArray[np.int64], npt.NDArray[np.int32]
]:
    """Distribute rows of a fixed-degree array to destination ranks.

    Uses a scalable neighbourhood exchange: send/receive ranks are
    discovered via NBX consensus rather than an all-to-all, keeping
    communication sparse as the communicator grows, at the cost of at
    least one non-blocking consensus round.

    Args:
        comm: MPI communicator that ``list``/``destinations`` are
            distributed across.
        list: Rows to distribute, with shape ``(num_nodes, degree)``.
            The global index of row ``i`` is assumed to be ``i`` plus
            the number of rows owned by lower-ranked processes.
        destinations: Destination rank(s) for the ith row of ``list``.
            The first rank is the 'owner' of the row; any further ranks
            receive it as a ghost.

    Returns:
        Tuple of (received rows, source rank of each received row,
        original global index of each received row, owning rank of the
        ghost rows). The last entry has one entry per ghost row -- the
        trailing rows of the first entry -- not one entry per received
        row.
    """
    return _cpp.graph.distribute(
        comm,
        np.ascontiguousarray(list, dtype=np.int64),
        destinations._cpp_object,  # type: ignore[arg-type]
    )


def comm_graph(map: IndexMap, root: int = 0) -> AdjacencyList:
    """Build a parallel communication graph from an index map.

    The communication graph is a directed graph that represents the
    communication pattern for a distributed array, and specifically the
    forward scatter operation where the values for owned indices are
    sent to ghosting ranks. The graph is built from an index map, which
    describes the local and ghosted indices of the array.

    Edges in the graph represent communication from the owning rank to
    ranks that ghost the data. The edge data holds the (0) target node,
    (1) edge weight, and (2) an indicator for whether the sending and
    receiving ranks share memory (``local==1``) or if the ranks do not
    share memory (``local==0``). The node data holds the local size
    (number of owned indices) and the number of ghost indices.

    The graph can be processed using :func:`comm_graph` to build data
    structures that can be used to build a `NetworkX
    <https://networkx.org/>`_ directed graph.

    Note:
        This function is collective across all MPI ranks. The
        communication graph is returned on the `root` rank. All other
        ranks return an empty graph

    Args:
        map: Index map to build the communication graph from.
        root: Rank that will return the communication graph. Other ranks
            return an empty graph.

    Returns:
        An adjacency list representing the communication graph.
    """
    return AdjacencyList(_cpp.graph.comm_graph(map._cpp_object, root))


def comm_graph_data(
    graph: AdjacencyList,
) -> tuple[list[tuple[int, int, dict[str, int]]], list[tuple[int, dict[str, int]]]]:
    """Build communication graph data for use with `NetworkX <https://networkx.org/>`_.

    Args:
        graph: Communication graph to build data from. Normally created
            by :func:`comm_graph`.

    Returns:
        A tuple of two lists. The first list contains the edge data,
        where an edge is a `(nodeID_0, nodeID_1, dict)` tuple, where
        `dict` holds edge data. The second list hold node data, where a
        node is a `(nodeID, dict)` tuple, where `dict` holds node data.
    """
    cpp_graph = graph._cpp_object
    if not isinstance(cpp_graph, _cpp.graph.AdjacencyList_int_sizet_int8__int32_int32):
        raise TypeError("comm_graph_data requires a graph created by comm_graph().")
    return _cpp.graph.comm_graph_data(cpp_graph)


def comm_to_json(graph: AdjacencyList) -> str:
    """Build and JSON string from a communication graph.

    The JSON string can be used to construct a `NetworkX
    <https://networkx.org/>`_ graph. This is helpful for cases where a
    simulation is executed and the graph data is written to file as a
    JSON string for later analysis.

    Args:
        graph: The communication graph to convert. Normally created by
            calling :meth:`comm_graph`.

    Returns:
        A JSON string representing the communication graph.
    """
    cpp_graph = graph._cpp_object
    if not isinstance(cpp_graph, _cpp.graph.AdjacencyList_int_sizet_int8__int32_int32):
        raise TypeError("comm_to_json requires a graph created by comm_graph().")
    return _cpp.graph.comm_to_json(cpp_graph)
