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
from dolfinx.cpp.fem import IntegralType, transpose_dofmap
from dolfinx.cpp.fem import (
    create_nonmatching_meshes_interpolation_data as _create_nonmatching_meshes_interpolation_data,
)
from dolfinx.cpp.fem import create_sparsity_pattern as _create_sparsity_pattern
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
    functionspace,
)
from dolfinx.mesh import Mesh as _Mesh


class PointOwnerShipData(typing.NamedTuple):
    """
    Convenience class for storing data related to the ownership of points.

    Attributes:
        src_owner: Ranks owning each point sent into ownership determination for current process
        dest_owners: Ranks that sent `dest_points` to current process
        dest_points: Points owned by current rank
        dest_cells: Cell indices (local to process) where each entry of `dest_points` is located
    """

    src_owner: npt.NDArray[np.int32]
    dest_owners: npt.NDArray[np.int32]
    dest_points: npt.NDArray[np.floating]
    dest_cells: npt.NDArray[np.int32]


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
) -> PointOwnerShipData:
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
        return PointOwnerShipData(
            *_create_nonmatching_meshes_interpolation_data(
                mesh_to._cpp_object, element, mesh_from._cpp_object, padding
            )
        )
    else:
        return PointOwnerShipData(
            *_create_nonmatching_meshes_interpolation_data(
                mesh_to, element, mesh_from._cpp_object, cells, padding
            )
        )


__all__ = [
    "Constant",
    "Expression",
    "Function",
    "ElementMetaData",
    "create_matrix",
    "functionspace",
    "FunctionSpace",
    "create_sparsity_pattern",
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
