"""Unit tests for the JIT compiler"""

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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2011-05-12
# Last changed: 2011-05-12

import unittest
from dolfin import *

class JIT(unittest.TestCase):

    def test_nasty_jit_caching_bug(self):

        # This may result in something like "matrices are not aligned"
        # from FIAT if the JIT caching does not recognize that the two
        # forms are different

        for representation in ["tensor", "quadrature"]:

            parameters["form_compiler"]["representation"] = representation

            M1 = assemble(Constant(1.0)*dx, mesh=UnitSquare(4, 4))
            M2 = assemble(Constant(1.0)*dx, mesh=UnitCube(4, 4, 4))

            self.assertAlmostEqual(M1, 1.0)
            self.assertAlmostEqual(M2, 1.0)

if __name__ == "__main__":
    print ""
    print "Testing JIT compiler"
    print "------------------------------------------------"
    unittest.main()
