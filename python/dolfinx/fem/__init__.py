# -*- coding: utf-8 -*-
# Copyright (C) 2017-2018 Garth N. Wells and Michal Habera
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
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
from dolfinx.fem.coordinatemap import create_coordinate_map
from dolfinx.fem.dirichletbc import (DirichletBC, locate_dofs_geometrical,
                                     locate_dofs_topological)
from dolfinx.fem.dofmap import DofMap
from dolfinx.fem.form import Form
from dolfinx.fem.formmanipulations import (adjoint, derivative, increase_order,
                                           tear)
from dolfinx.fem.function import (Constant, Expression, Function,
                                  FunctionSpace, TensorFunctionSpace,
                                  VectorFunctionSpace)
from dolfinx.fem.linearproblem import LinearProblem

__all__ = [
    "Constant", "Expression", "Function",
    "FunctionSpace", "TensorFunctionSpace",
    "VectorFunctionSpace",
    "create_vector", "create_vector_block", "create_vector_nest",
    "create_matrix", "create_matrix_block", "create_matrix_nest",
    "apply_lifting", "apply_lifting_nest", "assemble_scalar", "assemble_vector",
    "assemble_vector_block", "assemble_vector_nest",
    "assemble_matrix_block", "assemble_matrix_nest",
    "assemble_matrix", "set_bc", "set_bc_nest", "create_coordinate_map",
    "DirichletBC", "DofMap", "Form", "IntegralType",
    "derivative", "adjoint", "increase_order",
    "tear", "LinearProblem", "locate_dofs_geometrical", "locate_dofs_topological"
]
