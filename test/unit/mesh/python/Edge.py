"""Unit tests for the Edge class"""

__author__    = "Garth N. Wells (gnw20@cam.ac.uk)"
__date__      = "2011-02-26"
__copyright__ = "Copyright (C) 2011 Garth N. Wells"
__license__   = "GNU LGPL Version 2.1"

import unittest
from dolfin import *

cube   = UnitCube(5, 5, 5)
square = UnitSquare(5, 5)

class EdgeFunctions(unittest.TestCase):

    def test2DEdgeLength(self):
        """Iterate over edges and sum length."""
        length = 0.0
        for e in edges(square):
            length += e.length()
        self.assertAlmostEqual(length, 19.07106781186544708362)

    def test3DEdgeLength(self):
        """Iterate over edges and sum length."""
        length = 0.0
        for e in edges(cube):
            length += e.length()
        self.assertAlmostEqual(length, 278.58049080280125053832)

    def test3DEdgeDot(self):
        """Iterate over edges and sum length."""
        for e in edges(cube):
            dot = e.dot(e)/(e.length()**2)
            self.assertAlmostEqual(dot, 1.0)

    def test2DEdgeDot(self):
        """Iterate over edges and sum length."""
        for e in edges(square):
            dot = e.dot(e)/(e.length()**2)
            self.assertAlmostEqual(dot, 1.0)

if __name__ == "__main__":
    unittest.main()
