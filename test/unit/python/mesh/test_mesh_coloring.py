#!/usr/bin/env py.test

"""Unit tests for mesh coloring"""

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


def test_by_entity_cell_coloring():
    """Color mesh cells by connections."""

    # Get coloring libraries
    default_parameter = parameters["graph_coloring_library"]
    coloring_libraries = parameters.get_range("graph_coloring_library")
    for coloring_library in coloring_libraries:
        parameters["graph_coloring_library"] = coloring_library
        mesh = UnitCubeMesh(16, 16, 16)
        mesh.color("vertex")
        mesh.color("edge")
        mesh.color("facet")

    parameters["graph_coloring_library"] = default_parameter
