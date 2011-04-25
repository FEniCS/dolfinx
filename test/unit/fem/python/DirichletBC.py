"""Unit tests for assembly"""

__author__ = "Garth N. Wells (gnw20@cam.ac.uk)"
__date__ = "2011-04-25"
__copyright__ = "Copyright (C) 2011 Garth N. Wells"
__license__  = "GNU LGPL Version 2.1"

import unittest
import numpy
from dolfin import *

# FIXME: No complete


class DirichletBCTest(unittest.TestCase):

    def test_director_lifetime(self):
        """Test for any problems with objects with directors going out
        of scope"""

        class Boundary(SubDomain):
            def inside(self, x, on_boundary): return on_boundary
        class BoundaryFunction(Expression):
            def eval(self, values, x): values[0] = 1.0

        mesh = UnitSquare(8, 8)
        V = FunctionSpace(mesh, "Lagrange", 1)
        v, u = TestFunction(V), TrialFunction(V)
        A = assemble(v*u*dx)
        bc = DirichletBC(V, BoundaryFunction(), Boundary())
        bc.apply(A)

if __name__ == "__main__":
    print ""
    print "Testing basic DOLFIN DirichletBC operations"
    print "------------------------------------------------"
    unittest.main()
