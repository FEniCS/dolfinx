"""Unit tests for uBlas matrix and vector algebra"""

__author__ = "Garth N. Wells (g.n.wells@tudelft.nl)"
__date__ = "2006-08-11"
__copyright__ = "Copyright (C) 2006 Garth N. Wells"
__license__  = "GNU GPL Version 2"

import unittest
from dolfin import *

class AddVectors(unittest.TestCase):

    def testUBlasVector(self):
        """Add uBlas vectors"""
        x = Vector(10)
        y = Vector(10)
        z = Vector(10)
#        z = x + y
#        x(0) = 1.0
#        y(1) = 1.0    
#        z(0) = 1.0
#        z(1) = 1.0
#        self.assertEqual(x+y, z)

#class AddMatrices(unittest.TestCase):
#
#    def testUBlasVector(self):
#        """Add uBlas dense matrices"""
#        X = uBlasDenseMatrix(10,10)
#        Y = uBlasDenseMatrix(10,10)
#        Z = uBlasDenseMatrix(10,10)
#        X(2,3) = 1.0
#        Y(4,4) = 1.0    
#        Z(2,3) = 1.0
#        Z(4,4) = 1.0
#        self.assertEqual(X+Y, Z)

if __name__ == "__main__":
    unittest.main()
