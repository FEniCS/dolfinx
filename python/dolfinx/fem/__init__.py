# -*- coding: utf-8 -*-
# Copyright (C) 2017-2018 Garth N. Wells and Michal Habera
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tools for assembling and manipulating finite element forms"""

from dolfinx.fem.assemble import (create_vector, create_vector_block, create_vector_nest,
                                  create_matrix, create_matrix_block, create_matrix_nest,
                                  assemble_scalar,
                                  assemble_vector, assemble_vector_nest, assemble_vector_block,
                                  assemble_matrix, assemble_matrix_nest, assemble_matrix_block,
                                  set_bc, set_bc_nest,
                                  apply_lifting, apply_lifting_nest)
from dolfinx.fem.coordinatemapping import create_coordinate_map
from dolfinx.fem.dirichletbc import DirichletBC
from dolfinx.fem.dofmap import DofMap
from dolfinx.fem.form import Form
from dolfinx.cpp.fem import IntegralType
from dolfinx.fem.formmanipulations import (derivative, adjoint, increase_order,
                                           tear)
from dolfinx.fem.solving import solve
from dolfinx.fem.dirichletbc import locate_dofs_geometrical, locate_dofs_topological

__all__ = [
    "create_vector", "create_vector_block", "create_vector_nest",
    "create_matrix", "create_matrix_block", "create_matrix_nest",
    "apply_lifting", "apply_lifting_nest", "assemble_scalar", "assemble_vector",
    "assemble_vector_block", "assemble_vector_nest",
    "assemble_matrix_block", "assemble_matrix_nest",
    "assemble_matrix", "set_bc", "set_bc_nest", "create_coordinate_map",
    "DirichletBC", "DofMap", "Form", "IntegralType",
    "derivative", "adjoint", "increase_order",
    "tear", "project", "solve", "locate_dofs_geometrical", "locate_dofs_topological"
]
