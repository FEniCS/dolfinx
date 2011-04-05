"""Unit tests for adaptivity"""

__author__ = "Marie E. Rognes (meg@simula.no)"
__copyright__ = "Copyright (C) 2011 Marie E. Rognes"
__license__  = "GNU LGPL version 3 or any later version"

# First added:  2011-04-05
# Last changed: 2011-04-06

import unittest
from dolfin import *
from ufl.algorithms import replace

class ErrorControlTest(unittest.TestCase):

    def setUp(self):

        # Define variational problem
        mesh = UnitSquare(8, 8)
        V = FunctionSpace(mesh, "Lagrange", 1)
        bc = DirichletBC(V, 0.0, "x[0] < DOLFIN_EPS || x[0] > 1.0 - DOLFIN_EPS")

        u = TrialFunction(V)
        v = TestFunction(V)
        f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=1)
        g = Expression("sin(5*x[0])", degree=1)
        a = inner(grad(u), grad(v))*dx
        L = f*v*dx + g*v*ds
        problem = VariationalProblem(a, L, bc)

        # Define goal
        u = Function(V)
        M = u*dx
        self.goal = M

        # Generate error control forms
        ufl_forms = problem.generate_error_control_forms(u, M)

        # Compile generated forms
        forms = [Form(form) for form in ufl_forms]

        # Initialize error control object
        forms += [problem.is_linear]
        ec = cpp.ErrorControl(*forms)

        # Store created stuff
        self.problem = problem
        self.u = u
        self.ec = ec

    def test_error_estimation(self):

        # Solve variational problem once
        self.problem.solve(self.u)

        # Compute error estimate
        error_estimate = self.ec.estimate_error(self.u, self.problem.bcs)

        # Compare estimate with defined reference
        reference = 0.0011789985750808342
        self.assertAlmostEqual(error_estimate, reference)

    def test_error_indicators(self):

        # Solve variational problem once
        self.problem.solve(self.u)

        # Compute error indicators
        indicators = Vector(self.u.function_space().mesh().num_cells())
        indicators[0] = 1.0
        #self.ec.compute_indicators(indicators, self.u) #

        reference = 1.0 # FIXME
        self.assertAlmostEqual(indicators.sum(), reference)

    def test_adaptive_solve(self):

        # Solve problem adaptively
        self.problem.parameters["adaptivity"]["plot_mesh"] = False
        tol = 0.00087
        self.problem.solve(self.u, tol, self.goal)

        # Extract solution and update goal
        w = Function(self.u.fine().function_space())
        w.assign(self.u.fine())
        M = replace(self.goal, {self.u: w})

        # Compare computed goal with reference
        reference = 0.12583303389560166
        self.assertAlmostEqual(assemble(M), reference)

if __name__ == "__main__":
    print ""
    print "Testing automated adaptivity operations"
    print "------------------------------------------------"
    unittest.main()
