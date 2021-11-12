# -*- coding: utf-8 -*-
# Copyright (C) 2017-2018 Garth N. Wells and Michal Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tools for assembling and manipulating finite element forms"""

from dolfinx.cpp.fem import IntegralType
from dolfinx.fem.assemble import (apply_lifting, apply_lifting_nest,
                                  assemble_matrix, assemble_matrix_block,
                                  assemble_matrix_nest, assemble_scalar,
                                  assemble_vector, assemble_vector_block,
                                  assemble_vector_nest, create_matrix,
                                  create_matrix_block, create_matrix_nest,
                                  create_vector, create_vector_block,
                                  create_vector_nest, set_bc, set_bc_nest)
from dolfinx.fem.dirichletbc import (DirichletBC, bcs_by_block, locate_dofs_geometrical,
                                     locate_dofs_topological)
from dolfinx.fem.dofmap import DofMap
from dolfinx.fem.form import Form
from dolfinx.fem.formmanipulations import adjoint
from dolfinx.fem.function import (Constant, Expression, Function,
                                  FunctionSpace, TensorFunctionSpace,
                                  VectorFunctionSpace)
from dolfinx.fem.problem import LinearProblem, NonlinearProblem
import typing
from dolfinx.cpp.fem import (create_sparsity_pattern as _create_sparsity_pattern,
                             Form_complex128 as _FormComplex,
                             Form_float64 as _FormReal)


def create_sparsity_pattern(a: typing.Union[_FormComplex, _FormReal]):
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
    "create_vector", "create_vector_block", "create_vector_nest",
    "create_matrix", "create_matrix_block", "create_matrix_nest",
    "apply_lifting", "apply_lifting_nest", "assemble_scalar", "assemble_vector",
    "assemble_vector_block", "assemble_vector_nest",
    "assemble_matrix_block", "assemble_matrix_nest",
    "assemble_matrix", "set_bc", "set_bc_nest",
    "DirichletBC", "bcs_by_block", "DofMap", "Form", "IntegralType",
    "adjoint", "LinearProblem", "locate_dofs_geometrical", "locate_dofs_topological",
    "NonlinearProblem"]
