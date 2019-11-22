# -*- coding: utf-8 -*-
# Copyright (C) 2017-2018 Garth N. Wells and Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tools for assembling and manipulating finite element forms"""

from dolfin.fem.assemble import (create_vector, create_vector_block, create_vector_nest,
                                 create_matrix, create_matrix_block, create_matrix_nest,
                                 assemble_scalar,
                                 assemble_vector, assemble_vector_nest, assemble_vector_block,
                                 assemble_matrix, assemble_matrix_nest, assemble_matrix_block,
                                 set_bc, set_bc_nest,
                                 apply_lifting, apply_lifting_nest)
from dolfin.fem.coordinatemapping import create_coordinate_map
from dolfin.fem.dirichletbc import DirichletBC
from dolfin.fem.dofmap import DofMap
from dolfin.fem.form import Form
from dolfin.cpp.fem import FormIntegrals
from dolfin.fem.formmanipulations import (derivative, adjoint, increase_order,
                                          tear)
from dolfin.fem.solving import solve
from dolfin.fem.dirichletbc import locate_dofs_geometrical, locate_dofs_topological

__all__ = [
    "create_vector", "create_vector_block", "create_vector_nest",
    "create_matrix", "create_matrix_block", "create_matrix_nest",
    "apply_lifting", "apply_lifting_nest", "assemble_scalar", "assemble_vector",
    "assemble_vector_block", "assemble_vector_nest",
    "assemble_matrix_block", "assemble_matrix_nest",
    "assemble_matrix", "set_bc", "set_bc_nest", "create_coordinate_map",
    "DirichletBC", "DofMap", "Form", "FormIntegrals",
    "derivative", "adjoint", "increase_order",
    "tear", "project", "solve", "locate_dofs_geometrical", "locate_dofs_topological"
]
