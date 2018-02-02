"""Unit tests for assembly"""

# Copyright (C) 2018 Garth N. Wells
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

import pytest
import os
import numpy
import dolfin
from ufl import dx


def test_initialisation():
    mesh = dolfin.generation.UnitCubeMesh(dolfin.MPI.comm_world, 4, 4, 4)
    V = dolfin.function.functionspace.FunctionSpace(mesh, "Lagrange", 1)
    v = dolfin.function.argument.TestFunction(V)
    u = dolfin.function.argument.TrialFunction(V)
    f = dolfin.function.constant.Constant(0.0)
    a = v*u*dx
    L = v*f*dx

    assembler = dolfin.fem.assembling.Assembler(a, L)
