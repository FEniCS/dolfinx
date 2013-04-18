"""Unit tests for the Cell class"""

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
from dolfin import *

cube   = UnitCubeMesh(5, 5, 5)
square = UnitSquareMesh(5, 5)
meshes = [cube, square]

class IntervalTest(unittest.TestCase):

    def test_contains(self):
        "Test check for point in cell"

        mesh = UnitIntervalMesh(1)
        cell = Cell(mesh, 0)

        self.assertEqual(cell.contains(Point(0.5)), True)
        self.assertEqual(cell.contains(Point(1.5)), False)

class TriangleTest(unittest.TestCase):

    def test_contains(self):
        "Test check for point in cell"

        mesh = UnitSquareMesh(1, 1)
        cell = Cell(mesh, 0)

        self.assertEqual(cell.contains(Point(0.5)), True)
        self.assertEqual(cell.contains(Point(1.5)), False)

class TetrahedronTest(unittest.TestCase):

    def test_contains(self):
        "Test check for point in cell"

        mesh = UnitCubeMesh(1, 1, 1)
        cell = Cell(mesh, 0)

        self.assertEqual(cell.contains(Point(0.5)), True)
        self.assertEqual(cell.contains(Point(1.5)), False)

if __name__ == "__main__":
    unittest.main()
