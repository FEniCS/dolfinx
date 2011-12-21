"""Unit tests for the solve interface"""

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
# First added:  2011-12-21
# Last changed:se

import unittest
from dolfin import *

class solveTest(unittest.TestCase):

    def test_normalize(self):
        size = 200
        x = Vector(size)
        print "Sum", x.sum()
        x[:] = 2.0
        print "Sum", x.sum()
        cpp.normalize(x, "l2")
        print "Sum", x.norm("l2")
        cpp.normalize(x, "average")
        print "Sum", x.sum()


if __name__ == "__main__":
    # Turn of DOLFIN output
    set_log_active(False)

    print ""
    print "Testing DOLFIN la/solve interface"
    print "------------------------------------------------"
    unittest.main()
