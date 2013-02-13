"""Unit tests for LocalSolver"""

# Copyright (C) 2013 Garth N. Wells
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
# First added:  2013-02-13
# Last changed:

import unittest
import numpy
from dolfin import *

class LocalSolver(unittest.TestCase):

    def test_local_solve(self):

        mesh = UnitCubeMesh(16, 16, 16)
        V = FunctionSpace(mesh, "Lagrange", 2)

        v = TestFunction(V)
        u = TrialFunction(V)
        f = Constant(10.0)

        # Forms for projection
        a = inner(v, u)*dx
        L = inner(v, f)*dx

        # Wrap forms as DOLFIN forms (LocalSolver hasn't been properly
        # wrapped in Python yet)
        a = Form(a)
        L = Form(L)

        u = Function(V)
        local_solver = cpp.LocalSolver()
        local_solver.solve(u.vector(), a, L)
        x = u.vector().copy()
        x[:] = 10.0
        self.assertAlmostEqual((u.vector() - x).norm("l2"), 0.0, 10)


if __name__ == "__main__":
    print ""
    print "Testing class LocalSolver"
    print "-------------------------"
    unittest.main()
