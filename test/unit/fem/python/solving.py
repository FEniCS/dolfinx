"""Unit tests for the solve function"""

# Copyright (C) 2011 Anders Logg
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
# First added:  2011-11-10
# Last changed: 2011-11-10

import unittest
from dolfin import *

class Solving(unittest.TestCase):

    def test_bcs(self):
        "Check that the bcs argument is picked up"

        mesh = UnitSquare(4, 4)
        V = FunctionSpace(mesh, "Lagrange", 1)
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Constant(100.0)

        a = dot(grad(u), grad(v))*dx + u*v*dx
        L = f*v*dx

        bc = DirichletBC(V, 0.0, DomainBoundary())

        # Single bc argument
        u1 = Function(V)
        solve(a == L, u1, bc)

        # List of bcs
        u2 = Function(V)
        solve(a == L, u2, [bc])

        # Single bc keyword argument
        u3 = Function(V)
        solve(a == L, u3, bcs=bc)

        # List of bcs keyword argument
        u4 = Function(V)
        solve(a == L, u4, bcs=[bc])

        # Check all solutions
        self.assertAlmostEqual(u1.vector().norm("l2"), 14.9362601686, 10)
        self.assertAlmostEqual(u2.vector().norm("l2"), 14.9362601686, 10)
        self.assertAlmostEqual(u3.vector().norm("l2"), 14.9362601686, 10)
        self.assertAlmostEqual(u4.vector().norm("l2"), 14.9362601686, 10)

    def test_calling(self):
        "Test that unappropriate arguments are not allowed"
        mesh = UnitSquare(4, 4)
        V = FunctionSpace(mesh, "Lagrange", 1)
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Constant(100.0)

        a = dot(grad(u), grad(v))*dx + u*v*dx
        L = f*v*dx

        bc = DirichletBC(V, 0.0, DomainBoundary())

        kwargs = {"solver_parameters":{"linear_solver": "lu"},
                  "form_compiler_parameters":{"optimize": True}}

        A = assemble(a)
        b = assemble(L)
        x = Vector(len(b))

        def wrong_solve_0():
            solve(A, x, b, **kwargs)
        
        self.assertRaises(RuntimeError, wrong_solve_0)

        # FIXME: Include more tests for this versatile function

if __name__ == "__main__":
    print ""
    print "Testing the solve function"
    print "--------------------------"
    unittest.main()
