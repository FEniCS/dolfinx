# Copyright (C) 2024 Garth N. Wells and Paul T. Kühner
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
import ufl
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


class FiniteElement:
    _cpp_object: typing.Union[_cpp.fem.FiniteElement_float32, _cpp.fem.FiniteElement_float64]

    def __init__(self, cpp_object):
        self._cpp_object = cpp_object

    def __eq__(self, other):
        return self._cpp_object == other._cpp_object

    @property
    def dtype(self):
        return self._cpp_object.dtype

    @property
    def basix_element(self):
        return self._cpp_object.basix_element

    @property
    def num_sub_elements(self):
        return self._cpp_object.num_sub_elements

    @property
    def value_shape(self):
        return self._cpp_object.value_shape

    @property
    def interpolation_points(self):
        return self._cpp_object.interpolation_points

    @property
    def interpolation_ident(self):
        return self._cpp_object.interpolation_ident

    @property
    def space_dimension(self):
        return self._cpp_object.space_dimension

    @property
    def needs_dof_transformations(self):
        return self._cpp_object.needs_dof_transformations

    @property
    def signature(self):
        return self._cpp_object.signature

    def T_apply(self, x, cell_permutations, dim):
        self._cpp_object.T_apply(x, cell_permutations, dim)

    def Tt_apply(self, x, cell_permutations, dim):
        self._cpp_object.Tt_apply(x, cell_permutations, dim)

    def Tt_inv_apply(self, x, cell_permutations, dim):
        self._cpp_object.Tt_apply(x, cell_permutations, dim)


def finite_element(
    cell_type: _cpp.mesh.CellType,
    ufl_e: ufl.FiniteElementBase,
    dtype: np.dtype,
) -> FiniteElement:
    """Create a DOLFINx element from a basix.ufl element."""
    if np.issubdtype(dtype, np.float32):
        CppElement = _cpp.fem.FiniteElement_float32
    elif np.issubdtype(dtype, np.float64):
        CppElement = _cpp.fem.FiniteElement_float64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    if ufl_e.is_mixed:
        elements = [finite_element(cell_type, e, dtype) for e in ufl_e.sub_elements]
        return CppElement(elements)
    elif ufl_e.is_quadrature:
        return CppElement(
            cell_type, ufl_e.custom_quadrature()[0], ufl_e.reference_value_shape, ufl_e.is_symmetric
        )
    else:
        basix_e = ufl_e.basix_element._e
        value_shape = ufl_e.reference_value_shape if ufl_e.block_size > 1 else None
        return FiniteElement(CppElement(basix_e, value_shape, ufl_e.is_symmetric))
