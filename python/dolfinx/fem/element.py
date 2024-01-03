# Copyright (C) 2024 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Finite elements."""

from dolfinx import cpp as _cpp
import numpy as np
import basix


class CoordinateElement:
    """Coordinate element describing the geometry map for mesh cells"""

    def __init__(self, cmap):
        """Create a coordinate map element.

        Note:
            This initialiser is for internal use and not normally called
            in user code. Use :func:`coordinate_element` to create a CoordinateElement.

        Args:
            cmap: A C++ CoordinateElement.

        """
        self._cpp_object = cmap


def coordinate_element(celltype: basix.CellType, degree: int, dtype=np.float64,
                       variant=int(basix.LagrangeVariant.unset)):
    if dtype == np.float32:
        return CoordinateElement(_cpp.fem.CoordinateElement_float32(celltype, degree, variant))
    elif dtype == np.float64:
        return CoordinateElement(_cpp.fem.CoordinateElement_float64(celltype, degree, variant))
    else:
        raise RuntimeError("Unsupported dtype.")
