# Copyright (C) 2024 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Finite elements."""

import typing
from functools import singledispatch

import numpy as np
import numpy.typing as npt

import basix
from dolfinx import cpp as _cpp


class CoordinateElement:
    """Coordinate element describing the geometry map for mesh cells."""

    _cpp_object: typing.Union[
        _cpp.fem.CoordinateElement_float32, _cpp.fem.CoordinateElement_float64
    ]

    def __init__(
        self,
        cmap: typing.Union[_cpp.fem.CoordinateElement_float32, _cpp.fem.CoordinateElement_float64],
    ):
        """Create a coordinate map element.

        Note:
            This initialiser is for internal use and not normally called
            in user code. Use :func:`coordinate_element` to create a
            CoordinateElement.

        Args:
            cmap: A C++ CoordinateElement.
        """
        assert isinstance(
            cmap, (_cpp.fem.CoordinateElement_float32, _cpp.fem.CoordinateElement_float64)
        )
        self._cpp_object = cmap

    @property
    def dtype(self) -> np.dtype:
        """Scalar type for the coordinate element."""
        return np.dtype(self._cpp_object.dtype)

    @property
    def dim(self) -> int:
        """Dimension of the coordinate element space.

        This is number of basis functions that span the coordinate
        space, e.g., for a linear triangle cell the dimension will be 3.
        """
        return self._cpp_object.dim

    def create_dof_layout(self) -> _cpp.fem.ElementDofLayout:
        """Compute and return the dof layout"""
        return self._cpp_object.create_dof_layout()

    def push_forward(
        self,
        X: typing.Union[npt.NDArray[np.float32], npt.NDArray[np.float64]],
        cell_geometry: typing.Union[npt.NDArray[np.float32], npt.NDArray[np.float64]],
    ) -> typing.Union[npt.NDArray[np.float32], npt.NDArray[np.float64]]:
        """Push points on the reference cell forward to the physical cell.

        Args:
            X: Coordinates of points on the reference cell,
                ``shape=(num_points, topological_dimension)``.
            cell_geometry: Coordinate 'degrees-of-freedom' (nodes) of
                the cell, ``shape=(num_geometry_basis_functions,
                geometrical_dimension)``. Can be created by accessing
                ``geometry.x[geometry.dofmap.cell_dofs(i)]``,

        Returns:
            Physical coordinates of the points reference points ``X``.
        """
        return self._cpp_object.push_forward(X, cell_geometry)

    def pull_back(
        self,
        x: typing.Union[npt.NDArray[np.float32], npt.NDArray[np.float64]],
        cell_geometry: typing.Union[npt.NDArray[np.float32], npt.NDArray[np.float64]],
    ) -> typing.Union[npt.NDArray[np.float32], npt.NDArray[np.float64]]:
        """Pull points on the physical cell back to the reference cell.

        For non-affine cells, the pull-back is a nonlinear operation.

        Args:
            x: Physical coordinates to pull back to the reference cells,
                ``shape=(num_points, geometrical_dimension)``.
            cell_geometry: Physical coordinates describing the cell,
                shape ``(num_of_geometry_basis_functions, geometrical_dimension)``
                They can be created by accessing `geometry.x[geometry.dofmap.cell_dofs(i)]`,

        Returns:
            Reference coordinates of the physical points ``x``.
        """
        return self._cpp_object.pull_back(x, cell_geometry)

    @property
    def variant(self) -> int:
        """Lagrange variant of the coordinate element.

        Note:
            The return type is an integer. A Basix enum can be created using
            ``basix.LagrangeVariant(value)``.
        """
        return self._cpp_object.variant

    @property
    def degree(self) -> int:
        """Polynomial degree of the coordinate element."""
        return self._cpp_object.degree


@singledispatch
def coordinate_element(
    celltype: _cpp.mesh.CellType,
    degree: int,
    variant=int(basix.LagrangeVariant.unset),
    dtype: npt.DTypeLike = np.float64,
):
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
    if np.issubdtype(dtype, np.float32):
        return CoordinateElement(_cpp.fem.CoordinateElement_float32(celltype, degree, variant))
    elif np.issubdtype(dtype, np.float64):
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
