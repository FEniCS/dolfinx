#!/usr/bin/env py.test

"Unit tests for the MeshEditor class"

# Copyright (C) 2006-2011 Anders Logg
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
# First added:  2006-08-08
# Last changed: 2014-02-06

from dolfin import *


def test_triangle_mesh():

    # Create mesh object and open editor
    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, 2, 2)
    editor.init_vertices(3)  # test both versions of interface
    editor.init_vertices_global(3, 3)
    editor.init_cells(1)    # test both versions of interface
    editor.init_cells_global(1, 1)

    # Add vertices
    editor.add_vertex(0, 0.0, 0.0)
    editor.add_vertex(1, 1.0, 0.0)
    editor.add_vertex(2, 0.0, 1.0)

    # Add cell
    editor.add_cell(0, 0, 1, 2)

    # Close editor
    editor.close()
