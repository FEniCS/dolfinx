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
# Last changed: 2011-08-22

import unittest
import numpy
from dolfin import *

class MeshEditorTest(unittest.TestCase):

    def test_triangle_mesh(self):

        # Create mesh object and open editor
        mesh = Mesh()
        editor = MeshEditor()
        editor.open(mesh, 2, 2)
        editor.init_vertices(3)
        editor.init_cells(1)

        # Add vertices
        editor.add_vertex(0, 0.0, 0.0)
        editor.add_vertex(1, 1.0, 0.0)
        editor.add_vertex(2, 0.0, 1.0)

        # Add cell
        editor.add_cell(0, 0, 1, 2)

        # Close editor
        editor.close()

if __name__ == "__main__":
    unittest.main()
