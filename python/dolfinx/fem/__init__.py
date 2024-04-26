# Copyright (C) 2017-2018 Garth N. Wells and Michal Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tools for assembling and manipulating finite element forms."""

import typing

import numpy as np
import numpy.typing as npt

from dolfinx.cpp.fem import FiniteElement_float32 as _FiniteElement_float32
from dolfinx.cpp.fem import FiniteElement_float64 as _FiniteElement_float64
from dolfinx.cpp.fem import IntegralType, InterpolationType, transpose_dofmap
from dolfinx.cpp.fem import (
    create_nonmatching_meshes_interpolation_data as _create_nonmatching_meshes_interpolation_data,
)
from dolfinx.cpp.fem import create_sparsity_pattern as _create_sparsity_pattern
from dolfinx.cpp.fem import discrete_gradient as _discrete_gradient
from dolfinx.cpp.mesh import Geometry_float32 as _Geometry_float32
from dolfinx.cpp.mesh import Geometry_float64 as _Geometry_float64
from dolfinx.fem.assemble import (
    apply_lifting,
    assemble_matrix,
    assemble_scalar,
    assemble_vector,
    create_matrix,
    create_vector,
    set_bc,
)
from dolfinx.fem.bcs import (
    DirichletBC,
    bcs_by_block,
    dirichletbc,
    locate_dofs_geometrical,
    locate_dofs_topological,
)
from dolfinx.fem.dofmap import DofMap
from dolfinx.fem.element import CoordinateElement, coordinate_element
from dolfinx.fem.forms import Form, extract_function_spaces, form, form_cpp_class
from dolfinx.fem.function import (
    Constant,
    ElementMetaData,
    Expression,
    Function,
    FunctionSpace,
    PointOwnershipData,
    functionspace,
)
from dolfinx.la import MatrixCSR as _MatrixCSR
from dolfinx.mesh import Mesh as _Mesh


def create_sparsity_pattern(a: Form):
    """Create a sparsity pattern from a bilinear form.

    Args:
        a: Bilinear form to build a sparsity pattern for.

    Returns:
        Sparsity pattern for the form ``a``.

    Note:
        The pattern is not finalised, i.e. the caller is responsible for
        calling ``assemble`` on the sparsity pattern.

    """
    return _create_sparsity_pattern(a._cpp_object)


def create_nonmatching_meshes_interpolation_data(
    mesh_to: typing.Union[_Mesh, _Geometry_float64, _Geometry_float32],
    element: typing.Union[_FiniteElement_float32, _FiniteElement_float64],
    mesh_from: _Mesh,
    cells: typing.Optional[npt.NDArray[np.int32]] = None,
    padding: float = 1e-14,
) -> PointOwnershipData:
    """Generate data needed to interpolate discrete functions across different meshes.

    Args:
        mesh_to: Mesh or geometry of the mesh of the function space to interpolate into
        element: Element of the function space to interpolate into
        mesh_from: Mesh that the function to interpolate from is defined on
        cells: Indices of the cells in the destination mesh on which to interpolate.
        padding: Absolute padding of bounding boxes of all entities on mesh_to

    Returns:
        Data needed to interpolation functions defined on function spaces on the meshes.
    """
    if cells is None:
        return PointOwnershipData(
            *_create_nonmatching_meshes_interpolation_data(
                mesh_to._cpp_object, element, mesh_from._cpp_object, padding
            )
        )
    else:
        return PointOwnershipData(
            *_create_nonmatching_meshes_interpolation_data(
                mesh_to, element, mesh_from._cpp_object, cells, padding
            )
        )


def discrete_gradient(space0: FunctionSpace, space1: FunctionSpace) -> _MatrixCSR:
    """Assemble a discrete gradient operator.

    The discrete gradient operator interpolates the gradient of
    a H1 finite element function into a H(curl) space. It is assumed that
    the H1 space uses an identity map and the H(curl) space uses a covariant Piola map.

    Args:
        space0: H1 space to interpolate the gradient from
        space1: H(curl) space to interpolate into

    Returns:
        Discrete gradient operator
    """
    return _discrete_gradient(space0._cpp_object, space1._cpp_object)


__all__ = [
    "Constant",
    "Expression",
    "Function",
    "ElementMetaData",
    "create_matrix",
    "functionspace",
    "FunctionSpace",
    "create_sparsity_pattern",
    "discrete_gradient",
    "assemble_scalar",
    "assemble_matrix",
    "assemble_vector",
    "apply_lifting",
    "set_bc",
    "DirichletBC",
    "dirichletbc",
    "bcs_by_block",
    "DofMap",
    "Form",
    "form",
    "IntegralType",
    "InterpolationType",
    "create_vector",
    "locate_dofs_geometrical",
    "locate_dofs_topological",
    "extract_function_spaces",
    "transpose_dofmap",
    "create_nonmatching_meshes_interpolation_data",
    "CoordinateElement",
    "coordinate_element",
    "form_cpp_class",
]
