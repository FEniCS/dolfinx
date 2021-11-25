# Copyright (C) 2021 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Graph module"""

import numpy as np
from dolfinx import cpp as _cpp


def create_adjacencylist(data: np.ndarray, offsets=None):
    """Create an AdjacencyList"""
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
