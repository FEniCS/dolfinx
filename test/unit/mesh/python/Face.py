"""Unit tests for the Face class"""

__author__    = "Garth N. Wells (gnw20@cam.ac.uk)"
__date__      = "2011-02-26"
__copyright__ = "Copyright (C) 2011 Garth N. Wells"
__license__   = "GNU LGPL Version 2.1"

import unittest
from dolfin import *

cube = UnitCube(5, 5, 5)

class Area(unittest.TestCase):

    def testArea(self):
        """Iterate over faces and sum area."""
        area = 0.0
        for f in faces(cube):
            area += f.area()
        self.assertAlmostEqual(area, 39.21320343559672494393)

class Normal(unittest.TestCase):

    def testNormalPoint(self):
        """Compute normal vector to each face."""
        for f in faces(cube):
            n = f.normal()
            self.assertAlmostEqual(n.norm(), 1.0)

    def testNormalComponent(self):
        """Compute normal vector components to each face."""
        D = cube.topology().dim()
        for f in faces(cube):
            n = [f.normal(i) for i in range(D)]
            norm = sum(map(lambda x: x*x, n))
            self.assertAlmostEqual(norm, 1.0)

if __name__ == "__main__":
    unittest.main()
