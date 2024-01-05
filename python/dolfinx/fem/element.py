# Copyright (C) 2024 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Finite elements."""

import typing
from functools import singledispatch
import numpy.typing as npt

from dolfinx import cpp as _cpp
import numpy as np
import basix


class CoordinateElement:
    """Coordinate element describing the geometry map for mesh cells"""

    _cpp_object: typing.Union[_cpp.fem.CoordinateElement_float32, _cpp.fem.CoordinateElement_float64]

    def __init__(self, cmap):
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

    @property
    def dtype(self) -> npt.DTypeLike:
        """Scalar type for the coordinate element."""
        if isinstance(self._cpp_object, _cpp.fem.CoordinateElement_float32):
            return np.float32
        elif isinstance(self._cpp_object, _cpp.fem.CoordinateElement_float64):
            return np.float64
        else:
            raise RuntimeError("Unable to determine CoordinateElement scalar type.")


@singledispatch
def coordinate_element(celltype: typing.Any, degree: int,
                       variant=int(basix.LagrangeVariant.unset),
                       dtype: npt.DTypeLike = np.float64):
    """Create a Lagrange CoordinateElement from element metadata.

    Coordinate elements are typically used to create meshes.

    Args:
        celltype: Cell shape
        degree: Polynomial degree of the coordinate element map.
        variant: Basix Lagrange variant (affects node placement).
        dtype: Scalar type for the coordinate element.

    Returns:
        A coordinate element.
    """
    if dtype == np.float32:
        return CoordinateElement(_cpp.fem.CoordinateElement_float32(celltype, degree, variant))
    elif dtype == np.float64:
        return CoordinateElement(_cpp.fem.CoordinateElement_float64(celltype, degree, variant))
    else:
        raise RuntimeError("Unsupported dtype.")


@coordinate_element.register(basix.finite_element.FiniteElement)
def _(e: basix.finite_element.FiniteElement):
    """Create a Lagrange CoordinateElement from a Basix finite element.

    Coordinate elements are typically used when creating meshes.

    Args:
        e: Basix finite element.

    Returns:
        A coordinate element.
    """
    try:
        return CoordinateElement(_cpp.fem.CoordinateElement_float32(e._e))
    except TypeError:
        return CoordinateElement(_cpp.fem.CoordinateElement_float64(e._e))
