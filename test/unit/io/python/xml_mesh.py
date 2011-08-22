"""Unit tests for the XML io library for meshes"""

# Copyright (C) 2011 Garth N. Wells
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
# First added:  2011-06-17
# Last changed:

import unittest
from dolfin import *

class xml_mesh_io(unittest.TestCase):
    """Test output of Meshes to XML files"""

    def test_save_plain_mesh2D(self):
        mesh = UnitSquare(8, 8)
        f = File("unit_square.xml")
        f << mesh

    def test_save_plain_mesh3D(self):
        mesh = UnitCube(8, 8, 8)
        f = File("unit_cube.xml")
        f << mesh


if __name__ == "__main__":
    unittest.main()
