"""Unit tests for uBlas matrix and vector algebra"""

__author__ = "Garth N. Wells (g.n.wells@tudelft.nl)"
__date__ = "2006-08-11"
__copyright__ = "Copyright (C) 2006 Garth N. Wells"
__license__  = "GNU GPL Version 2"

import unittest
from dolfin import *

class SetGetVectors(unittest.TestCase):

    def testUBlasVectorSet(self):
        """Set values of uBlas"""
        x = uBlasVector(3)
        x.set(0,1)
        x.set(1,1)
        x.set(2,1)
#       Can we get DOLFIN_EPS?
        self.assertTrue( abs(x.norm()-3.0**(0.5) ) < 1.0e-8 )

    def testUBlasVectorGet(self):
        """Set values of uBlas"""
        x = uBlasVector(3)
        x.set(0, 1)
        x.set(1, 1)
        x.set(2, 1)
        sumx = x.get(0) + x.get(1) + x.get(2)
        self.assertTrue( abs(sumx-3.0 ) < 1.0e-8 )

#class AddVectors(unittest.TestCase):
#
#    def testUBlasVector(self):
#        """Add uBlas vectors"""
#        x = uBlasVector(10)
#        x.disp()
#        x.set(1, 1)
#        x.disp()
#        y = uBlasVector(10)
#        z = uBlasVector(10)
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
