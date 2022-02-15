# -*- coding: utf-8 -*-
# Copyright (C) 2017-2018 Garth N. Wells and Michal Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tools for assembling and manipulating finite element forms."""

from dolfinx.cpp.fem import transpose_dofmap  # noqa
from dolfinx.cpp.fem import IntegralType
from dolfinx.cpp.fem import create_sparsity_pattern as _create_sparsity_pattern
from dolfinx.fem import petsc
from dolfinx.fem.assemble import (apply_lifting, assemble_matrix,
                                  assemble_scalar, assemble_vector, set_bc)
from dolfinx.fem.bcs import (DirichletBCMetaClass, bcs_by_block, dirichletbc,
                             locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.fem.dofmap import DofMap
from dolfinx.fem.formmanipulations import adjoint
from dolfinx.fem.forms import FormMetaClass, extract_function_spaces, form
from dolfinx.fem.function import (Constant, Expression, Function,
                                  FunctionSpace, TensorFunctionSpace,
                                  VectorFunctionSpace)
from dolfinx.fem.problem import LinearProblem, NonlinearProblem


def create_sparsity_pattern(a: FormMetaClass):
    """Create a sparsity pattern from a bilinear form"""
    topology = a.mesh.topology
    dofmap0 = a.function_spaces[0].dofmap
    dofmap1 = a.function_spaces[1].dofmap
    types = a.integral_types
    return _create_sparsity_pattern(topology, [dofmap0, dofmap1], types)


__all__ = [
    "Constant", "Expression", "Function",
    "FunctionSpace", "TensorFunctionSpace",
    "VectorFunctionSpace",
    "assemble_scalar", "assemble_matrix", "assemble_vector", "apply_lifting", "set_bc",
    "DirichletBCMetaClass", "dirichletbc", "bcs_by_block", "DofMap", "FormMetaClass", "form", "IntegralType",
    "adjoint", "LinearProblem", "locate_dofs_geometrical", "locate_dofs_topological",
    "NonlinearProblem", "extract_function_spaces", "petsc"]
