"""Unit tests for graph building"""

# Copyright (C) 2013 Garth N. Wells
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
#
# First added:  2013-08-10
# Last changed:

import pytest
from dolfin import *


def test_build_from_mesh_simple():
    """Build mesh graph """

    mesh = UnitCubeMesh(16, 16, 16)
    D = mesh.topology().dim()
    GraphBuilder.local_graph(mesh, D, 0)
    GraphBuilder.local_graph(mesh, D, 1)
    GraphBuilder.local_graph(mesh, 2, D)
    GraphBuilder.local_graph(mesh, 1, D)
    GraphBuilder.local_graph(mesh, 0, D)
