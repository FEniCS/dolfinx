# -*- coding: utf-8 -*-
# Copyright (C) 2017-2018 Garth N. Wells and Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tools for assembling and manipulating finite element forms"""

from dolfin.fem.assemble import (assemble, assemble_vector, assemble_matrix,
                                 set_bc)
from dolfin.fem.assembling import (assemble_local, assemble_system,
                                   SystemAssembler)
from dolfin.fem.coordinatemapping import create_coordinate_map
from dolfin.fem.dirichletbc import DirichletBC
from dolfin.fem.dofmap import DofMap
from dolfin.fem.form import Form
from dolfin.fem.formmanipulations import (derivative, adjoint, increase_order,
                                          tear)
from dolfin.fem.interpolation import interpolate
from dolfin.fem.projection import project
from dolfin.fem.solving import solve

__all__ = [
    "assemble_local", "assemble_system", "assemble", "assemble_vector",
    "assemble_matrix", "set_bc", "SystemAssembler", "create_coordinate_map",
    "DirichletBC", "DofMap", "Form", "derivative", "adjoint", "increase_order",
    "tear", "interpolate", "project", "solve"
]
