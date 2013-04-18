"""Unit tests for intersection computation"""

# Copyright (C) 2013 Anders Logg
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
# First added:  2013-04-18
# Last changed: 2013-04-18

import unittest

from dolfin import MeshPointIntersection
from dolfin import UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh
from dolfin import Point

class BoundingBoxTreeTest(unittest.TestCase):

    def test_mesh_point_1d(self):
        "Test mesh-point intersection in 1D"

        point = Point(0.1)
        mesh = UnitIntervalMesh(16)

        intersection = MeshPointIntersection(mesh, point)

        self.assertEqual(intersection.intersected_cells(), [1])

    def test_mesh_point_2d(self):
        "Test mesh-point intersection in 2D"

        point = Point(0.1, 0.2)
        mesh = UnitSquareMesh(16, 16)

        intersection = MeshPointIntersection(mesh, point)

        self.assertEqual(intersection.intersected_cells(), [98])

    def test_mesh_point_3d(self):
        "Test mesh-point intersection in 3D"

        point = Point(0.1, 0.2, 0.3)
        mesh = UnitCubeMesh(8, 8, 8)

        intersection = MeshPointIntersection(mesh, point)

        self.assertEqual(intersection.intersected_cells(), [816])

if __name__ == "__main__":
    print ""
    print "Testing intersection computation"
    print "------------------------------------------------"
    unittest.main()
