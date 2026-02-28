# Copyright (C) 2021-2024 Garth N. Wells and Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Graph representations and operations on graphs."""

from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

from dolfinx import cpp as _cpp
from dolfinx.cpp.graph import partitioner

# Import graph partitioners, which may or may not be available
# (dependent on build configuration)
try:
    from dolfinx.cpp.graph import partitioner_scotch  # noqa
except ImportError:
    pass
try:
    from dolfinx.cpp.graph import partitioner_parmetis  # noqa
except ImportError:
    pass
try:
    from dolfinx.cpp.graph import partitioner_kahip  # noqa
except ImportError:
    pass


__all__ = [
    "AdjacencyList",
    "adjacencylist",
    "comm_graph",
    "comm_graph_data",
    "comm_to_json",
    "partitioner",
]


_T = TypeVar("_T", np.int32, np.int64)


class AdjacencyList(Generic[_T]):
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

    def __repr__(self):
        """String representation of the adjacency list."""
        return self._cpp_object.__repr__()

    def links(self, node: int) -> npt.NDArray[_T]:
        """Retrieve the links of a node.

        Note:
            This is available only for adjacency lists with no
            additional link (edge) data.

        Args:
            node: Node to retrieve the connectivity of.

        Returns:
            Neighbors of the node.
        """
        return self._cpp_object.links(node)

    @property
    def array(self) -> npt.NDArray[_T]:
        """Array representation of the adjacency list.

        Note:
            This is available only for adjacency lists with no
            additional link (edge) data.

        Returns:
            Flattened array representation of the adjacency list.
        """
        return self._cpp_object.array

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
        return self._cpp_object.num_nodes


def adjacencylist(
    data: npt.NDArray[_T], offsets: npt.NDArray[np.int32] | None = None
) -> AdjacencyList[_T]:
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
    if data.dtype == np.int32:
        cpp_t = _cpp.graph.AdjacencyList_int32
    elif data.dtype == np.int64:
        cpp_t = _cpp.graph.AdjacencyList_int64
    else:
        raise TypeError("Data type for adjacency list not supported.")

    cpp_object = cpp_t(data, offsets) if offsets is not None else cpp_t(data)
    return AdjacencyList(cpp_object)


def comm_graph(map: _cpp.common.IndexMap, root: int = 0) -> AdjacencyList:
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
    return AdjacencyList(_cpp.graph.comm_graph(map))


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
    return _cpp.graph.comm_graph_data(graph._cpp_object)


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
    return _cpp.graph.comm_to_json(graph._cpp_object)
