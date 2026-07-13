# Copyright (C) 2017-2018 Garth N. Wells and Michal Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tools for assembling and manipulating finite element forms.

Note:
    ``dolfinx.fem.petsc`` and ``dolfinx.fem.problems`` require optional
    dependencies and must be explicitly imported.
"""

from dolfinx.cpp.fem import _IntegralType as IntegralType
from dolfinx.cpp.fem import transpose_dofmap
from dolfinx.fem.assemble import (
    apply_lifting,
    assemble_matrix,
    assemble_scalar,
    assemble_vector,
    create_matrix,
    create_vector,
    pack_coefficients,
    pack_constants,
)
from dolfinx.fem.bcs import (
    DirichletBC,
    bcs_by_block,
    dirichletbc,
    locate_dofs_geometrical,
    locate_dofs_topological,
)
from dolfinx.fem.dofmap import DofMap, create_dofmaps
from dolfinx.fem.element import CoordinateElement, FiniteElement, coordinate_element, finiteelement
from dolfinx.fem.forms import (
    Form,
    compile_form,
    create_form,
    extract_function_spaces,
    form,
    form_cpp_class,
    mixed_topology_form,
)
from dolfinx.fem.function import (
    Constant,
    ElementMetaData,
    Expression,
    Function,
    FunctionSpace,
    functionspace,
)
from dolfinx.fem.utils import (
    build_sparsity_pattern,
    compute_integration_domains,
    create_interpolation_data,
    create_sparsity_pattern,
    discrete_curl,
    discrete_gradient,
    interpolate_geometry,
    interpolation_matrix,
)

__all__ = [
    "Constant",
    "CoordinateElement",
    "DirichletBC",
    "DofMap",
    "ElementMetaData",
    "Expression",
    "FiniteElement",
    "Form",
    "Function",
    "FunctionSpace",
    "IntegralType",
    "apply_lifting",
    "assemble_matrix",
    "assemble_scalar",
    "assemble_vector",
    "bcs_by_block",
    "build_sparsity_pattern",
    "compile_form",
    "compute_integration_domains",
    "coordinate_element",
    "create_dofmaps",
    "create_form",
    "create_interpolation_data",
    "create_matrix",
    "create_sparsity_pattern",
    "create_vector",
    "dirichletbc",
    "discrete_curl",
    "discrete_gradient",
    "extract_function_spaces",
    "finiteelement",
    "form",
    "form_cpp_class",
    "functionspace",
    "interpolate_geometry",
    "interpolation_matrix",
    "locate_dofs_geometrical",
    "locate_dofs_topological",
    "mixed_topology_form",
    "pack_coefficients",
    "pack_constants",
    "set_bc",
    "transpose_dofmap",
]
