"Unit tests for the mesh library"

# Copyright (C) 2012 Anders Logg
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
# First added:  2012-01-16
# Last changed: 2012-01-16

import unittest
from dolfin import *

class MeshTransformation(unittest.TestCase):

    def test_rotation_2d(self):
        mesh = UnitSquareMesh(8, 8)
        p = Point(1, 2)
        mesh.rotate(10)
        mesh.rotate(10, 2)
        mesh.rotate(10, 2, p)

    def test_rotation_3d(self):
        mesh = UnitCubeMesh(8, 8, 8)
        p = Point(1, 2, 3)
        mesh.rotate(30, 0)
        mesh.rotate(30, 1)
        mesh.rotate(30, 2)
        mesh.rotate(30, 0, p)

if __name__ == "__main__":
    unittest.main()
