"""This module contains unit tests for functionaliy only available in
Python. This functionality is implemented in site-packages/dolfin"""

__author__ = "Anders Logg <logg@simula.no>"
__date__ = "2009-11-16 -- 2009-11-16"
__copyright__ = "Copyright (C) 2009 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

import unittest
from dolfin import *

class ErrorNorm(unittest.TestCase):

    def test_error_norm(self):

        # Approximation (zero)
        mesh = UnitSquare(4, 4)
        V = FunctionSpace(mesh, "CG", 2)
        u_h = Function(V)
        u_h.vector().zero()

        # Exact solution
        u = Expression("x[0]*x[0]", element=V.ufl_element())

        # Norm of error
        e = errornorm(u_h, u)

        self.assertAlmostEqual(e, sqrt(1.0 / 5.0))

if __name__ == "__main__":
    print ""
    print "Testing Python extras"
    print "----------------------------------------------------------------------"
    unittest.main()
