"""Unit test for the mesh library"""

import unittest
from dolfin import *

class SimpleShapes(unittest.TestCase):

    def testUnitSquare(self):
        """Create mesh of unit square"""
        mesh = NewUnitSquare(5, 7)
        self.assertEqual(mesh.numVertices(), 48)
        self.assertEqual(mesh.numCells(), 70)

    def testUnitCube(self):
        """Create mesh of unit cube"""
        mesh = NewUnitCube(5, 7, 9)
        self.assertEqual(mesh.numVertices(), 480)
        self.assertEqual(mesh.numCells(), 1890)

if __name__ == "__main__":
    unittest.main()
