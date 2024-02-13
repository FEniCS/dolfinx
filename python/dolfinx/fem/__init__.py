# Copyright (C) 2017-2018 Garth N. Wells and Michal Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tools for assembling and manipulating finite element forms."""

from dolfinx.cpp.fem import (
    IntegralType,
    create_nonmatching_meshes_interpolation_data,
    transpose_dofmap,
)
from dolfinx.cpp.fem import create_sparsity_pattern as _create_sparsity_pattern
from dolfinx.cpp.fem import discrete_gradient as _discrete_gradient
from dolfinx.cpp.fem import transpose_dofmap
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


def discrete_gradient(space0: FunctionSpace, space1: FunctionSpace) -> _MatrixCSR:
    """Assemble a discrete gradient operator.

    The discrete gradient operator A interpolates the gradient of
    a Lagrange finite element function into a Nedelec (first kind)
    space.

    Args:
        space0: Lagrange space to interpolate the gradient from
        space1: Nedelec space to interpolate into

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
