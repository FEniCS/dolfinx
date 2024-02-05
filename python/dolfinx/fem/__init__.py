# Copyright (C) 2017-2018 Garth N. Wells and Michal Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tools for assembling and manipulating finite element forms."""

from dolfinx.cpp.fem import IntegralType
from dolfinx.cpp.fem import create_nonmatching_meshes_interpolation_data as _create_nonmatching_meshes_interpolation_data
from dolfinx.cpp.fem import create_sparsity_pattern as _create_sparsity_pattern
from dolfinx.cpp.fem import transpose_dofmap
from dolfinx.fem.assemble import (apply_lifting, assemble_matrix, assemble_scalar, assemble_vector, create_matrix,
                                  create_vector, set_bc)
from dolfinx.fem.bcs import DirichletBC, bcs_by_block, dirichletbc, locate_dofs_geometrical, locate_dofs_topological
from dolfinx.fem.dofmap import DofMap
from dolfinx.fem.element import CoordinateElement, coordinate_element
from dolfinx.fem.forms import Form, extract_function_spaces, form, form_cpp_class
from dolfinx.fem.function import Constant, ElementMetaData, Expression, Function, FunctionSpace, functionspace


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
    mesh_to: typing.Union(Mesh, Geometry), element: FiniteElement, mesh_from: Mesh,
    cells: typing.Optional[npt.NDArray[np.int32]] = None, padding: float = 1e-14
):
    """Generate data needed to interpolate discrete functions across different meshes.

    Args:
        mesh_to: Mesh or geometry to interpolate into
        element: Element of the space to interpolate into
        mesh_from: Mesh of the function to interpolate from
        cells: Indices of the cells in the destination mesh on which to interpolate.
        padding: Absolute padding of bounding boxes of all entities on mesh_to
    """
    if cells is None:
        assert isinstance(mesh_to, Mesh)
        _create_nonmatching_meshes_interpolation_data(
            mesh_to._cpp_object, element, mesh_from._cpp_onject, padding)
    else:
        assert isinstance(mesh_to, Geometry)
        _create_nonmatching_meshes_interpolation_data(
            mesh_to._cpp_object, element, mesh_from._cpp_object, cells, padding)


__all__ = [
    "Constant", "Expression", "Function", "ElementMetaData", "create_matrix",
    "functionspace", "FunctionSpace", "create_sparsity_pattern",
    "assemble_scalar", "assemble_matrix", "assemble_vector", "apply_lifting", "set_bc",
    "DirichletBC", "dirichletbc", "bcs_by_block", "DofMap", "Form",
    "form", "IntegralType", "create_vector",
    "locate_dofs_geometrical", "locate_dofs_topological",
    "extract_function_spaces", "transpose_dofmap", "create_nonmatching_meshes_interpolation_data",
    "CoordinateElement", "coordinate_element", "form_cpp_class"]
