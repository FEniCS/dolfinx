"""Unit tests for assembly"""

# Copyright (C) 2011 Garth N. Wells
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
# First added:  2011-04-25
# Last changed: 2011-04-25

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

    def test_near_and_between(self):
        "Test that near and between handle rounding errors correctly."
        # TODO: Move this test to a test module 'basic' or something?
        from dolfin.cpp import near, between # TODO: Import from dolfin directly
        from dolfin import DOLFIN_EPS as eps
        # Loop over magnitudes
        for i in range(100):
            # Loop over base values
            for j in range(1, 10):
                # Compute a value v and some values close to it
                v = j*10**(i-15)
                vm = v - eps
                vp = v + eps

                # Check that we return True when we should by definition:
                self.assertTrue(near(v, vm))
                self.assertTrue(near(v, vp))
                self.assertTrue(between(vm, v, vp))

                # vm and vp can round off to v, make some small values != v
                v2m = v * (1.0 - 2*eps) - 2*eps
                v2p = v * (1.0 + 2*eps) + 2*eps

                # Close to 1 except for some of the smallest v's:
                self.assertTrue(v/v2m > 1.0)
                self.assertTrue(v/v2p < 1.0)
                self.assertTrue(between(v2m, v, v2p))

                # Check that we can fail for fairly close values
                self.assertFalse(near(v, v2m))
                self.assertFalse(near(v, v2p))
                self.assertFalse(between(v2p, v, v2m))


if __name__ == "__main__":
    print ""
    print "Testing basic DOLFIN DirichletBC operations"
    print "------------------------------------------------"
    unittest.main()
