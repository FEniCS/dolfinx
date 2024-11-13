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


__all__ = ["AdjacencyList", "adjacencylist", "partitioner"]


class AdjacencyList:
    _cpp_object: Union[_cpp.la.AdjacencyList_int32, _cpp.la.AdjacencyList_int64]

    def __init__(self, cpp_object: Union[_cpp.la.AdjacencyList_int32, _cpp.la.AdjacencyList_int64]):
        self._cpp_object = cpp_object

    def links(self, node: Union[np.int32, np.int64]) -> npt.NDArray[Union[np.int32, np.int64]]:
        return self._cpp_object.links(node)

    @property
    def array(self) -> npt.NDArray[Union[np.int32, np.int64]]:
        return self._cpp_object.array

    @property
    def offsets(self) -> npt.NDArray[np.int32]:
        return self._cpp_object.offsets

    @property
    def num_nodes(self) -> np.int32:
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
    # Switch to np.isdtype(data.dtype, np.int32) once numpy >= 2.0 is enforced
    is_32bit = data.dtype == np.int32
    cpp_t = _cpp.graph.AdjacencyList_int32 if is_32bit else _cpp.graph.AdjacencyList_int64
    cpp_object = cpp_t(data, offsets) if offsets is not None else cpp_t(data)
    return AdjacencyList(cpp_object)
