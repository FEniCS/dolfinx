"""This module contains unit tests for functionaliy only available in
Python. This functionality is implemented in site-packages/dolfin"""

# Copyright (C) 2009 Anders Logg
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
# First added:  2009-11-16
# Last changed: 2009-11-16

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
        e = errornorm(u, u_h)

        self.assertAlmostEqual(e, sqrt(1.0/5.0))

if __name__ == "__main__":
    print ""
    print "Testing Python extras"
    print "----------------------------------------------------------------------"
    unittest.main()
