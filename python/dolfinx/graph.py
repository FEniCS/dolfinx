# Copyright (C) 2021 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Graph representations and operations on graphs."""

from __future__ import annotations

import typing

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


__all__ = ["adjacencylist", "partitioner"]


def adjacencylist(
    data: typing.Union[npt.NDArray[np.int32], npt.NDArray[np.int64]],
    offsets: typing.Optional[npt.NDArray[np.int32]] = None,
):
    """Create an AdjacencyList for int32 or int64 datasets.

    Args:
        data: The adjacency array. If the array is one-dimensional,
            offsets should be supplied. If the array is two-dimensional
            the number of edges per node is the second dimension.
        offsets: The offsets array with the number of edges per node.

    Returns:
        An adjacency list.

    """
    if offsets is None:
        try:
            return _cpp.graph.AdjacencyList_int32(data)
        except TypeError:
            return _cpp.graph.AdjacencyList_int64(data)
    else:
        try:
            return _cpp.graph.AdjacencyList_int32(data, offsets)
        except TypeError:
            return _cpp.graph.AdjacencyList_int64(data, offsets)
