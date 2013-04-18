"""Unit tests for BoundingBoxTree"""

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
# First added:  2013-04-15
# Last changed: 2013-04-18

import unittest

from dolfin import BoundingBoxTree
from dolfin import UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh
from dolfin import Point

class BoundingBoxTreeTest(unittest.TestCase):

    def test_unit_interval(self):
        "Test basic creation and point location for unit interval"

        reference = {1: [4]}

        p = Point(0.3)
        mesh = UnitIntervalMesh(16)

        for dim in range(1, 2):
            bbtree = BoundingBoxTree(mesh, dim)
            entities = bbtree.find(p)
            self.assertEqual(sorted(entities), reference[dim])

    def test_unit_square(self):
        "Test basic creation and point location for unit square"

        reference = {1: [226],
                     2: [136, 137]}

        p = Point(0.3, 0.3)
        mesh = UnitSquareMesh(16, 16)

        for dim in range(1, 3):
            bbtree = BoundingBoxTree(mesh, dim)
            entities = bbtree.find(p)
            self.assertEqual(sorted(entities), reference[dim])

    def test_unit_cube(self):
        "Test basic creation and point location for unit cube"

        reference = {1: [1364],
                     2: [1967, 1968, 1970, 1972, 1974, 1976],
                     3: [876, 877, 878, 879, 880, 881]}

        p = Point(0.3, 0.3, 0.3)
        mesh = UnitCubeMesh(8, 8, 8)

        for dim in range(1, 4):
            bbtree = BoundingBoxTree(mesh, dim)
            entities = bbtree.find(p)
            self.assertEqual(sorted(entities), reference[dim])

if __name__ == "__main__":
    print ""
    print "Testing BoudingBoxTree"
    print "------------------------------------------------"
    unittest.main()
