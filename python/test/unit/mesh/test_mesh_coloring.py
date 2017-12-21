"""Unit tests for mesh coloring"""

# Copyright (C) 2013-2016 Garth N. Wells
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
from dolfin import *
from dolfin_utils.test import pushpop_parameters

def test_by_entity_cell_coloring(pushpop_parameters):
    """Color mesh cells by connections."""

    # Get coloring libraries
    coloring_libraries = parameters.get_range("graph_coloring_library")
    for coloring_library in coloring_libraries:
        parameters["graph_coloring_library"] = coloring_library
        mesh = UnitCubeMesh(16, 16, 16)
        mesh.color("vertex")
        mesh.color("edge")
        mesh.color("facet")

        # Compute facet-based coloring with distance 2
        dim = mesh.topology().dim()
        coloring_type = (dim, dim - 1, dim, dim - 1, dim)
        mesh.color(coloring_type);
        colors = MeshColoring.cell_colors(mesh, coloring_type)
