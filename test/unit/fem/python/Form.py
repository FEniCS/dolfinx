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

class FormTest(unittest.TestCase):

    def setUp(self):
        self.mesh = UnitSquareMesh(10, 10)
        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.f = Expression("sin(pi*x[0]*x[1])")
        self.v = TestFunction(self.V)
        self.u = TrialFunction(self.V)

    def test_assemble(self):
        ufl_form = self.f*self.u*self.v*dx
        dolfin_form = Form(ufl_form)
        ufc_form = dolfin_form._compiled_form
        A_ufl_norm = assemble(ufl_form).norm("frobenius")
        A_dolfin_norm = assemble(dolfin_form).norm("frobenius")
        A_ufc_norm = assemble(ufc_form, coefficients=[self.f],
                              function_spaces=[self.V, self.V]).norm("frobenius")

        self.assertAlmostEqual(A_ufl_norm, A_dolfin_norm)
        self.assertAlmostEqual(A_ufl_norm, A_ufc_norm)

class FormTestsOverManifolds(unittest.TestCase):

    def setUp(self):

        # 1D in 2D spaces
        self.mesh1 = BoundaryMesh(UnitSquareMesh(2, 2))
        self.V1 = FunctionSpace(self.mesh1, "CG", 1)
        self.Q1 = FunctionSpace(self.mesh1, "DG", 0)

        # 2D in 3D spaces
        self.mesh2 = BoundaryMesh(UnitCubeMesh(2, 2, 2))
        self.V2 = FunctionSpace(self.mesh2, "CG", 1)
        self.Q2 = FunctionSpace(self.mesh2, "DG", 0)
        self.VV2 = VectorFunctionSpace(self.mesh2, "CG", 1)
        self.W2 = FunctionSpace(self.mesh2, "RT", 1)

    def test_assemble_functional(self):
        u = Function(self.V1)
        u.vector()[:] = 1.0
        surfacearea = assemble(u*dx)
        self.assertAlmostEqual(surfacearea, 4.0)

        u = Function(self.V2)
        u.vector()[:] = 1.0
        surfacearea = assemble(u*dx)
        self.assertAlmostEqual(surfacearea, 6.0)

        f = Expression(("1.0", "0.0", "0.0"))
        u = interpolate(f, self.VV2)
        print assemble(inner(u, u)*dx)
        u = interpolate(f, self.W2)
        print assemble(inner(u, u)*dx)

        f = Expression("1.0")
        u = interpolate(f, self.V1)
        surfacearea = assemble(u*dx)
        self.assertAlmostEqual(surfacearea, 4.0)

        f = Expression("1.0")
        u = interpolate(f, self.V2)
        surfacearea = assemble(u*dx)
        self.assertAlmostEqual(surfacearea, 6.0)


    def test_assemble_form(self):
        u = Function(self.V1)
        w = TestFunction(self.Q1)
        u.vector()[:] = 0.5
        facetareas = assemble(u*w*dx).array().sum()
        self.assertAlmostEqual(facetareas, 2.0)

        u = Function(self.V2)
        w = TestFunction(self.Q2)
        u.vector()[:] = 0.5
        a = u*w*dx
        b = assemble(a)
        facetareas = assemble(u*w*dx).array().sum()
        self.assertAlmostEqual(facetareas, 3.0)




if __name__ == "__main__":
    print ""
    print "Testing PyDOLFIN Form operations"
    print "------------------------------------------------"
    unittest.main()
