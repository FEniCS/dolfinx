"""Unit tests for the RKSolver interface"""

# Copyright (C) 2013 Johan Hake
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
# First added:  2013-02-20
# Last changed: 2013-02-20

import unittest
from dolfin import *

import numpy as np
from dolfin.fem.butcherscheme import _butcher_scheme_generator

class RKSolverTest(unittest.TestCase):

    def test_butcher_scalar_scheme(self):
        mesh = UnitSquareMesh(4, 4)
        V = FunctionSpace(mesh, "R", 0)
        u = Function(V)
        v = TestFunction(V)
        form = u*v*dx
        
        for Scheme in [ForwardEuler, BackwardEuler]:
            scheme = Scheme(form, u)
            solver = RKSolver(scheme)
        
if __name__ == "__main__":
    print ""
    print "Testing PyDOLFIN RKSolver operations"
    print "------------------------------------------------"
    unittest.main()
