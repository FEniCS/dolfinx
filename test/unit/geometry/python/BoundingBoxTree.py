"""Unit test for BoundingBoxTree"""

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
# Last changed: 2013-04-15

import unittest

from dolfin import BoundingBoxTree
from dolfin import UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh
from dolfin import Point

class BoundingBoxTreeTest(unittest.TestCase):

    def test_unit_interval(self):

        mesh = UnitIntervalMesh(16)
        for dim in range(2):
            tree = BoundingBoxTree(mesh, dim)

    def test_unit_square(self):

        mesh = UnitSquareMesh(16, 16)
        for dim in range(3):
            tree = BoundingBoxTree(mesh, dim)

    def test_unit_cube(self):

        mesh = UnitCubeMesh(8, 8, 8)
        for dim in range(4):
            tree = BoundingBoxTree(mesh, dim)

if __name__ == "__main__":
    print ""
    print "Testing BoudingBoxTree"
    print "------------------------------------------------"

    # FIXME: Temporary while testing
    mesh = UnitCubeMesh(3, 3, 3)
    tree = BoundingBoxTree(mesh)

    p = Point(0.5, 0.5, 0.5)
    entities = tree.find(p)

    p = Point(0.1, 0.1, 0.1)
    entities = tree.find(p)

    print entities

    #unittest.main()
