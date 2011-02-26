"""Unit tests for the Edge class"""

__author__    = "Garth N. Wells (gnw20@cam.ac.uk)"
__date__      = "2011-02-26"
__copyright__ = "Copyright (C) 2011 Garth N. Wells"
__license__   = "GNU LGPL Version 2.1"

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
