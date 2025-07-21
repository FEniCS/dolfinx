# Copyright (C) 2021-2024 Garth N. Wells and Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Graph representations and operations on graphs."""

from __future__ import annotations

from typing import Optional, Union

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


class AdjacencyList:
    _cpp_object: Union[
        _cpp.la.AdjacencyList_int32,
        _cpp.la.AdjacencyList_int64,
        _cpp.graph.AdjacencyList_int_sizet_int8__int32_int32,
    ]

    def __init__(
        self,
        cpp_object: Union[
            _cpp.graph.AdjacencyList_int32,
            _cpp.graph.AdjacencyList_int64,
            _cpp.graph.AdjacencyList_int_sizet_int8__int32_int32,
        ],
    ):
        """Creates a Python wrapper for the exported adjacency list class.

        Note:
            Do not use this constructor directly. Instead use
            :func:`adjacencylist`.

        Args:
            The underlying cpp instance that this object will wrap.
        """
        self._cpp_object = cpp_object

    def __repr__(self):
        return self._cpp_object.__repr__

    def links(self, node: Union[np.int32, np.int64]) -> npt.NDArray[Union[np.int32, np.int64]]:
        """Retrieve the links of a node.

        Args:
            Node to retrieve the connectivity of.

        Returns:
            Neighbors of the node.
        """
        return self._cpp_object.links(node)

    @property
    def array(self) -> npt.NDArray[Union[np.int32, np.int64]]:
        """Array representation of the adjacency list.

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
    data: npt.NDArray[Union[np.int32, np.int64]], offsets: Optional[npt.NDArray[np.int32]] = None
) -> AdjacencyList:
    """Create an AdjacencyList for int32 or int64 datasets.

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
    communication pattern of a distributed array. The graph is built
    from an index map, which describes the local and ghosted indices
    of the array.

    Edges in the graph represent communication from the owning rank to
    the ranks that ghost the data. The edge data holds the (0) target
    node, (1) edge weight, and (2) an indicator for whether the
    receiving rank shares memory with the caller (``local==1``) or is a
    remote memory location (``local==0``). The node data holds the local
    size (number of owned indices) and the number of ghost indices.

    The graph can be processed using :meth:`comm_graph` to build data
    that can be used to build a `NetworkX <https://networkx.org/>`_
    directed graph.

    Note:
        This function is collective across all MPI ranks. The graph is
        returned on the `root` rank. All other ranks return an empty graph

    Args:
        map: An index map tp build the communication graph from.
        root: The rank that will return the graph. If the graph is empty,
            it will be returned on all ranks.

    Returns:
        An adjacency list representing the communication graph.
    """
    return AdjacencyList(_cpp.graph.comm_graph(map))


def comm_graph_data(
    graph: AdjacencyList,
) -> tuple[list[tuple[int, int, dict[str, int]]], list[tuple[int, dict[str, int]]]]:
    """Convert a communication graph to data for use with
    `NetworkX <https://networkx.org/>`_.

    Args:
        graph: The communication graph to convert. Normally created by
            calling :meth:`comm_graph`.

    Returns:
        A tuple of two lists. The first list contains the edge data,
        where edge is a `(nodeID_0, nodeID_1, dict)` tuple, where `dict`
        holds edge data. The second list hold node data, where a node is
        a `(nodeID, dict)` tuple, where `dict` holds node data.
    """
    return _cpp.graph.comm_graph_data(graph._cpp_object)


def comm_to_json(graph: AdjacencyList) -> str:
    """Convert a communication graph to a JSON string.

    The JSON string can be used to construct a `NetworkX
    <https://networkx.org/>`_graph.

    Args:
        graph: The communication graph to convert. Normally created by
            calling :meth:`comm_graph`.

    Returns:
        A JSON string representing the communication graph.
    """
    return _cpp.graph.comm_to_json(graph._cpp_object)
