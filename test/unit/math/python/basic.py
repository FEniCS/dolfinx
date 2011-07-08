"""Unit tests for basic math functions"""

# Copyright (C) 2011 Martin Alnaes
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
# Modified by Garth N. Wells, 2011
#
# First added:  2011-07-108
# Last changed:

import unittest
import numpy
from dolfin import *

class DirichletBCTest(unittest.TestCase):

    def test_near(self):
        eps = DOLFIN_EPS
        # Loop over magnitudes
        for i in range(100):
            # Loop over base values
            for j in range(1, 10):
                # Compute a value v and some values close to it
                v = j*10**(i-13)
                #print "number:", v
                vm = v - eps
                vp = v + eps
                #vm = v - v*eps # Scaling eps does not work
                #vp = v + v*eps

                # Check that we return True when we should by definition:
                self.assertTrue(near(v, v))
                self.assertTrue(near(vm, vm))
                self.assertTrue(near(vp, vp))
                #self.assertTrue(near(v, vm)) # Can fail 
                #self.assertTrue(near(v, vp))
                if not near(v, vm):
                    print "not near vm: %r, %r" % (v, vm)
                if not near(v, vp):
                    print "not near vp: %r, %r" % (v, vp)

                # vm and vp can round off to v, make some small values != v
                # that are close to 1 (except for some of the smallest v's)
                v2m = v * (1.0 - 2*eps) - 2*eps
                v2p = v * (1.0 + 2*eps) + 2*eps
                self.assertTrue(v/v2m > 1.0)
                self.assertTrue(v/v2p < 1.0)

                # Check that we can fail for fairly close values
                self.assertFalse(near(v, v2m))
                self.assertFalse(near(v, v2p))

    def test_between(self):
        eps = DOLFIN_EPS
        # Loop over magnitudes
        for i in range(100):
            # Loop over base values
            for j in range(1, 10):
                # Compute a value v and some values close to it
                v = j*10**(i - 15)
                vm = v - eps
                vp = v + eps

                # Check that we return True when we should by definition:
                self.assertTrue(between(vm, v, vp))

                # vm and vp can round off to v, make some small values != v
                v2m = v * (1.0 - 2*eps) - 2*eps
                v2p = v * (1.0 + 2*eps) + 2*eps

                # Close to 1 except for some of the smallest v's:
                self.assertTrue(v/v2m > 1.0)
                self.assertTrue(v/v2p < 1.0)
                self.assertTrue(between(v2m, v, v2p))

                # Check that we can fail for fairly close values
                self.assertFalse(between(v2p, v, v2m))


if __name__ == "__main__":
    print ""
    print "Testing basic DOLFIN maths operations"
    print "------------------------------------------------"
    unittest.main()
