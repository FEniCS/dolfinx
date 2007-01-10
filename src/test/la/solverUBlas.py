"""Unit tests for uBlas linear solvers"""

__author__ = "Garth N. Wells (g.n.wells@tudelft.nl)"
__date__ = "2006-08-11"
__copyright__ = "Copyright (C) 2006 Garth N. Wells"
__license__  = "GNU GPL Version 2"

import unittest
from dolfin import *

class CreateSolvers(unittest.TestCase):

    def testUBlasKrylovSolver(self):
        """Create uBlas Krylov solver"""
        solver = uBlasKrylovSolver()

#    def testUBlasKrylovSolverNamed(self):
#        """Create uBlas Krylov solver with named method"""
#        solver = uBlasKrylovSolver(uBlasKrylovSolver::gmres)

#    def testUBlasKrylovSolverNamedPreconditioner(self):
#        """Create uBlas Krylov solver with named preconditioner"""
#        solver = uBlasKrylovSolver(uBlasPreconditioner::gmres)

#    def testUBlasKrylovSolverPreconditioner(self):
#        """Create uBlas Krylov solver with preconditioner"""
#        preconditioner = uBlasDummyPreconditioner()        
#        solver = uBlasKrylovSolver(preconditioner)

    def testUBlasLUSolver(self):
        """Create uBlas LU solver"""
        solver = uBlasLUSolver()

if __name__ == "__main__":
    unittest.main()
