"""Unit tests for the Edge class"""

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
# First added:  2011-02-26
# Last changed: 2011-02-26

import unittest
from dolfin import *

cube   = UnitCube(5, 5, 5)
square = UnitSquare(5, 5)
meshes = [cube, square]

class EdgeFunctions(unittest.TestCase):

    def test2DEdgeLength(self):
        """Iterate over edges and sum length."""
        length = 0.0
        for e in edges(square):
            length += e.length()
        if MPI.num_processes() == 1:
            self.assertAlmostEqual(length, 19.07106781186544708362)

    def test3DEdgeLength(self):
        """Iterate over edges and sum length."""
        length = 0.0
        for e in edges(cube):
            length += e.length()
        if MPI.num_processes() == 1:
            self.assertAlmostEqual(length, 278.58049080280125053832)

    def testEdgeDot(self):
        """Iterate over edges compute dot product with self."""
        for mesh in meshes:
            for e in edges(mesh):
                dot = e.dot(e)/(e.length()**2)
                self.assertAlmostEqual(dot, 1.0)

if __name__ == "__main__":
    unittest.main()
