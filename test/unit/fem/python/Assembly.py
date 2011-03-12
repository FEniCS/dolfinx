"""Unit tests for assembly"""

__author__ = "Garth N. Wells (gnw20@cam.ac.uk)"
__date__ = "2011-03-12"
__copyright__ = "Copyright (C) 2011 Garth N. Wells"
__license__  = "GNU LGPL Version 2.1"

import unittest
import numpy
from dolfin import *

class Assembly(unittest.TestCase):

    def test_functional_assmebly(self):

        mesh = UnitSquare(24, 24)
        f = Constant(1.0)

        M = f*dx
        parameters["num_threads"] = 0
        self.assertAlmostEqual(assemble(M, mesh=mesh), 1.0)

        M = f*ds
        parameters["num_threads"] = 0
        self.assertAlmostEqual(assemble(M, mesh=mesh), 4.0)

if __name__ == "__main__":
    print ""
    print "Testing basic DOLFIN assembly operations"
    print "------------------------------------------------"
    unittest.main()
