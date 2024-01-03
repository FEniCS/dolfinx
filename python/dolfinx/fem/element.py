# Copyright (C) 2024 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Finite elements."""

from functools import singledispatch

from dolfinx import cpp as _cpp
import numpy as np
import basix


class CoordinateElement:
    """Coordinate element describing the geometry map for mesh cells"""

    def __init__(self, cmap, dtype=None):
        """Create a coordinate map element.

        Note:
            This initialiser is for internal use and not normally called
            in user code. Use :func:`coordinate_element` to create a
            CoordinateElement.

        Args:
            cmap: A C++ CoordinateElement.

        """
        assert isinstance(cmap, (_cpp.fem.CoordinateElement_float32, _cpp.fem.CoordinateElement_float64))
        self._cpp_object = cmap


@singledispatch
def coordinate_element(celltype: basix.CellType, degree: int, dtype=np.float64,
                       variant=int(basix.LagrangeVariant.unset)):
    """Create a Lagrange CoordinateElement form element metadata.

    Coordinate elements are typically used when creating meshes.

    Args:
        celltype: Cell shape
        degree: Polynomial degree of the element map
        dtype: Scalar type
        variant: Basix Lagrange variant (affects node placement)

    Returns:
        A coordinate element.
    """
    if dtype == np.float32:
        return CoordinateElement(_cpp.fem.CoordinateElement_float32(celltype, degree, variant))
    elif dtype == np.float64:
        return CoordinateElement(_cpp.fem.CoordinateElement_float64(celltype, degree, variant))
    else:
        raise RuntimeError("Unsupported dtype.")


@coordinate_element.register
def _(e: basix._basixcpp.FiniteElement, dtype):
    """Create a Lagrange CoordinateElement from a Basix finite element.

    Coordinate elements are typically used when creating meshes.

    Args:
        e: Basix finite element
        dtype: Scalar type

    Returns:
        A coordinate element.
    """
    if dtype == np.float32:
        return CoordinateElement(_cpp.fem.CoordinateElement_float32(e))
    elif dtype == np.float64:
        return CoordinateElement(_cpp.fem.CoordinateElement_float64(e))
    else:
        raise RuntimeError("Unsupported dtype.")
