# Copyright (C) 2017-2018 Garth N. Wells and Michal Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tools for assembling and manipulating finite element forms."""

import numpy as np
import numpy.typing as npt

from dolfinx.cpp.fem import IntegralType, transpose_dofmap
from dolfinx.cpp.fem import create_interpolation_data as _create_interpolation_data
from dolfinx.cpp.fem import create_sparsity_pattern as _create_sparsity_pattern
from dolfinx.cpp.fem import discrete_gradient as _discrete_gradient
from dolfinx.cpp.fem import interpolation_matrix as _interpolation_matrix
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
from dolfinx.fem.forms import (
    Form,
    compile_form,
    create_form,
    extract_function_spaces,
    form,
    form_cpp_class,
)
from dolfinx.fem.function import (
    Constant,
    ElementMetaData,
    Expression,
    Function,
    FunctionSpace,
    functionspace,
)
from dolfinx.geometry import PointOwnershipData as _PointOwnershipData
from dolfinx.la import MatrixCSR as _MatrixCSR


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


def create_interpolation_data(
    V_to: FunctionSpace,
    V_from: FunctionSpace,
    cells: npt.NDArray[np.int32],
    padding: float = 1e-14,
) -> _PointOwnershipData:
    """Generate data needed to interpolate discrete functions across different meshes.

    Args:
        V_to: Function space to interpolate into
        V_from: Function space to interpolate from
        cells: Indices of the cells associated with `V_to` on which to
            interpolate into.
        padding: Absolute padding of bounding boxes of all entities on
            mesh_to

    Returns:
        Data needed to interpolation functions defined on function
        spaces on the meshes.
    """
    return _PointOwnershipData(
        _create_interpolation_data(
            V_to.mesh._cpp_object.geometry, V_to.element, V_from.mesh._cpp_object, cells, padding
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


def interpolation_matrix(space0: FunctionSpace, space1: FunctionSpace) -> _MatrixCSR:
    """Assemble an interpolation matrix for two function spaces on the same mesh.

    Args:
        space0: space to interpolate from
        space1: space to interpolate into

    Returns:
        Interpolation matrix
    """
    return _MatrixCSR(_interpolation_matrix(space0._cpp_object, space1._cpp_object))


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
    "create_vector",
    "locate_dofs_geometrical",
    "locate_dofs_topological",
    "extract_function_spaces",
    "transpose_dofmap",
    "create_interpolation_data",
    "CoordinateElement",
    "coordinate_element",
    "form_cpp_class",
    "create_form",
    "compile_form",
]
