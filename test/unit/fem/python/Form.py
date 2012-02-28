"""Unit tests for the fem interface"""

# Copyright (C) 2011 Johan Hake
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
# First added:  2011-12-02
# Last changed: 2011-12-02

import unittest
import numpy
from dolfin import *


mesh = UnitSquare(10, 10)
V = FunctionSpace(mesh, "CG", 1)
f = Expression("sin(pi*x[0]*x[1])")
v = TestFunction(V)
u = TrialFunction(V)

class FormTest(unittest.TestCase):

    def test_assemble(self):
        
        ufl_form = f*u*v*dx
        dolfin_form = Form(ufl_form)
        ufc_form = dolfin_form._compiled_form
        A_ufl_norm = assemble(ufl_form).norm("frobenius")
        A_dolfin_norm = assemble(dolfin_form).norm("frobenius")
        A_ufc_norm = assemble(ufc_form, coefficients=[f],
                              function_spaces=[V, V]).norm("frobenius")
        
        self.assertAlmostEqual(A_ufl_norm, A_dolfin_norm)
        self.assertAlmostEqual(A_ufl_norm, A_ufc_norm)

if __name__ == "__main__":
    print ""
    print "Testing PyDOLFIN Form operations"
    print "------------------------------------------------"
    unittest.main()
