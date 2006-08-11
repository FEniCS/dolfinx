"""Unit test for the mesh library"""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2006-08-08 -- 2006-08-09"
__copyright__ = "Copyright (C) 2006 Anders Logg"
__license__  = "GNU GPL Version 2"

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

class MeshRefinement(unittest.TestCase):

    def testRefineUnitSquare(self):
        """Refine mesh of unit square"""
        mesh = NewUnitSquare(5, 7)
        mesh.refine()
        self.assertEqual(mesh.numVertices(), 165)
        self.assertEqual(mesh.numCells(), 280)

    def testRefineUnitCube(self):
        """Refine mesh of unit cube"""
        mesh = NewUnitCube(5, 7, 9)
        mesh.refine()
        self.assertEqual(mesh.numVertices(), 3135)
        self.assertEqual(mesh.numCells(), 15120)

if __name__ == "__main__":
    unittest.main()
