"""Unit tests for adaptivity"""

# Copyright (C) 2011 Marie E. Rognes
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2011-04-05
# Last changed: 2011-04-06

import unittest
#from unittest import skipIf # Awaiting Python 2.7
from dolfin import *
from ufl.algorithms import replace

#@skipIf("Skipping error control test in parallel", MPI.num_processes() > 1)
class ErrorControlTest(unittest.TestCase):

    def setUp(self):

        # Define variational problem
        mesh = UnitSquare(8, 8)
        V = FunctionSpace(mesh, "Lagrange", 1)
        bc = DirichletBC(V, 0.0, "x[0] < DOLFIN_EPS || x[0] > 1.0 - DOLFIN_EPS")

        u = TrialFunction(V)
        v = TestFunction(V)
        f = Expression("10*std::exp(-(std::pow(x[0] - 0.5, 2) + std::pow(x[1] - 0.5, 2)) / 0.02)", degree=1)
        g = Expression("std::sin(5*x[0])", degree=1)
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

        if MPI.num_processes() > 1:
            return

        # Solve variational problem once
        self.problem.solve(self.u)

        # Compute error estimate
        error_estimate = self.ec.estimate_error(self.u, self.problem.bcs)

        # Compare estimate with defined reference
        reference = 0.0011789985750808342
        self.assertAlmostEqual(error_estimate, reference)

    def test_error_indicators(self):

        if MPI.num_processes() > 1:
            return

        # Solve variational problem once
        self.problem.solve(self.u)

        # Compute error indicators
        indicators = Vector(self.u.function_space().mesh().num_cells())
        indicators[0] = 1.0
        #self.ec.compute_indicators(indicators, self.u) #

        reference = 1.0 # FIXME
        self.assertAlmostEqual(indicators.sum(), reference)

    def test_adaptive_solve(self):

        if MPI.num_processes() > 1:
            return

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
