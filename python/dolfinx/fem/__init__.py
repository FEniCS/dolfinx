# Copyright (C) 2017-2018 Garth N. Wells and Michal Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tools for assembling and manipulating finite element forms."""

from dolfinx.cpp.fem import transpose_dofmap  # noqa
from dolfinx.cpp.fem import (IntegralType,
                             create_nonmatching_meshes_interpolation_data)
from dolfinx.cpp.fem import create_sparsity_pattern as _create_sparsity_pattern
from dolfinx.fem.assemble import (apply_lifting, assemble_matrix,
                                  assemble_scalar, assemble_vector,
                                  create_matrix, set_bc, create_vector)
from dolfinx.fem.bcs import (DirichletBC, bcs_by_block, dirichletbc,
                             locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.fem.dofmap import DofMap
from dolfinx.fem.forms import Form, extract_function_spaces, form
from dolfinx.fem.function import (Constant, Expression, Function, ElementMetaData,
                                  FunctionSpace, FunctionSpaceBase, VectorFunctionSpace)


def create_sparsity_pattern(a: Form):
    """Create a sparsity pattern from a bilinear form.

    Args:
        a: The bilinear form to build a sparsity pattern for.

    Returns:
        Sparsity pattern for the form ``a``.

    """
    return _create_sparsity_pattern(a._cpp_object)


__all__ = [
    "Constant", "Expression", "Function", "ElementMetaData", "create_matrix",
    "FunctionSpace", "FunctionSpaceBase", "VectorFunctionSpace",
    "create_sparsity_pattern",
    "assemble_scalar", "assemble_matrix", "assemble_vector", "apply_lifting", "set_bc",
    "DirichletBC", "dirichletbc", "bcs_by_block", "DofMap", "Form",
    "form", "IntegralType", "create_vector",
    "locate_dofs_geometrical", "locate_dofs_topological",
    "extract_function_spaces", "create_nonmatching_meshes_interpolation_data"]
